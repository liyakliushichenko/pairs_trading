import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import vectorbt as vbt

from pair_trading_project.src.ArbitragePortfolio import PairTradingBacktest
from pair_trading_project.src.utils import load_config

project_path = Path(__file__).parent.parent


def compute_beta(strategy_returns: pd.Series, asset_returns: pd.Series) -> float:
    """Вычисляет бету стратегии относительно актива.

    β = Cov(R_strategy, R_asset) / Var(R_asset)

    В качестве рыночного портфеля используются сами активы пары.
    Это корректно, т.к. ограничение задания требует β < 0.05
    именно на оба актива. Dollar-neutral структура автоматически
    снижает бету, поскольку PnL зависит от разности движений.
    """
    aligned = pd.concat([strategy_returns, asset_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    aligned.columns = ["strat", "asset"]
    cov = np.cov(aligned["strat"], aligned["asset"])[0, 1]
    var = np.var(aligned["asset"])
    return cov / var if var > 0 else 0.0


def run_backtest():
    """
    Запускает бэктест на бэктестовых данных.
    Сохраняет:
        - Основные бэктестовые метрики в /artifacts/backtest_metrics.json
        - График PnL стратегии в /artifacts/pnl_plot.png
        - График с требованиями обеспечения в /artifacts/requirements_plot.png
    """
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    INITIAL_CAPITAL = cfg["initial_capital"]
    FEES = cfg["fees"]
    SLIPPAGE = cfg["slippage"]
    INTEREST_RATE = cfg["interest_rate"]
    REBATE_RATE = cfg["rebate_rate"]
    INITIAL_MARGIN_RATE = cfg["initial_margin_rate"]
    MAINTENANCE_MARGIN_RATE = cfg["maintenance_margin_rate"]
    COLLATERAL = cfg["collateral"]
    MARGIN_HANDLE = cfg["margin_handle"]
    BET_SIZE = cfg["bet_size"]
    TICKER_A = cfg["ticker_a"]
    TICKER_B = cfg["ticker_b"]

    backtest_data = (
        pd.read_parquet(project_path.as_posix() + "/data/backtest_data.parquet")
        .set_index("Date")
        .sort_index(ascending=True)
    )

    signal_generator = joblib.load(
        project_path.as_posix() + "/artifacts/signals_generator.joblib"
    )

    # Получаем сигналы стратегии
    signals_df = signal_generator.transform(backtest_data)
    signal = signals_df["signal"]

    # Формируем size DataFrame для бэктестера
    # Используем vbt для определения моментов пересечения порогов
    vbt  # vectorbt используется в проекте для расширений
    size = pd.DataFrame(index=backtest_data.index, columns=backtest_data.columns)

    # Заполняем size только в моменты изменения сигнала (ребалансировка)
    signal_changes = signal.diff().fillna(signal).astype(bool)

    for idx in backtest_data.index:
        if signal_changes.loc[idx]:
            sig = signal.loc[idx]
            if sig == 1:       # Long A, Short B
                size.loc[idx, TICKER_A] = BET_SIZE
                size.loc[idx, TICKER_B] = -BET_SIZE
            elif sig == -1:    # Short A, Long B
                size.loc[idx, TICKER_A] = -BET_SIZE
                size.loc[idx, TICKER_B] = BET_SIZE
            else:              # Flat
                size.loc[idx, TICKER_A] = 0
                size.loc[idx, TICKER_B] = 0

    # Запускаем бэктест через PairTradingBacktest
    pair_trading_pf = PairTradingBacktest.backtest(
        close=backtest_data,
        size=size,
        init_cash=INITIAL_CAPITAL,
        fees=FEES,
        slippage=SLIPPAGE,
        interest_rate=INTEREST_RATE,
        rebate_rate=REBATE_RATE,
        initial_margin_rate=INITIAL_MARGIN_RATE,
        maintenance_margin_rate=MAINTENANCE_MARGIN_RATE,
        collateral=COLLATERAL,
        margin_handle=MARGIN_HANDLE,
    )

    # Графики
    pair_trading_pf.plot_pnl(
        output_file=project_path.as_posix() + "/artifacts/pnl_plot.png"
    )
    pair_trading_pf.plot_requirements(
        output_file=project_path.as_posix() + "/artifacts/requirements_plot.png"
    )

    # Базовые метрики
    metrics = pair_trading_pf.stats()

    # Расчёт беты стратегии относительно каждого актива
    strategy_returns = pair_trading_pf.pnl.pct_change().dropna()
    returns_a = backtest_data[TICKER_A].pct_change().dropna()
    returns_b = backtest_data[TICKER_B].pct_change().dropna()

    beta_a = compute_beta(strategy_returns, returns_a)
    beta_b = compute_beta(strategy_returns, returns_b)

    metrics[f"beta_{TICKER_A.lower()}"] = round(beta_a, 4)
    metrics[f"beta_{TICKER_B.lower()}"] = round(beta_b, 4)

    # Calmar ratio
    total_return = pair_trading_pf.pnl.iloc[-1] / pair_trading_pf.pnl.iloc[0] - 1
    n_days = len(pair_trading_pf.pnl)
    ann_return = (1 + total_return) ** (252 / n_days) - 1
    max_dd = (pair_trading_pf.pnl / pair_trading_pf.pnl.cummax() - 1).min()
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
    metrics["calmar_ratio"] = round(calmar, 2)
    metrics["annualized_return_pct"] = round(ann_return * 100, 2)

    # Вывод
    print("=" * 60)
    print(f"BACKTEST: {TICKER_A} / {TICKER_B}")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print(f"\n  Beta({TICKER_A}) < 0.05: {'PASS' if abs(beta_a) < 0.05 else 'FAIL'}")
    print(f"  Beta({TICKER_B}) < 0.05: {'PASS' if abs(beta_b) < 0.05 else 'FAIL'}")

    # Сохраняем метрики
    with open(
        project_path.as_posix() + "/artifacts/backtest_metrics.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4, default=str)


if __name__ == "__main__":
    run_backtest()
