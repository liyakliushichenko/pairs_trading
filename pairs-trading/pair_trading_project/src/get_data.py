import os
import time
from pathlib import Path

import pandas as pd
import requests

from pair_trading_project.src.utils import load_config

project_path = Path(__file__).parent.parent


def download_moex_candles(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Скачивает дневные свечи с MOEX ISS API (TQBR).

    Args:
        ticker: Тикер на MOEX (например 'LKOH').
        start: Дата начала в формате 'YYYY-MM-DD'.
        end: Дата окончания в формате 'YYYY-MM-DD'.

    Returns:
        DataFrame с колонками Date и тикер (close price).
    """
    all_rows = []
    cursor_start = 0
    base_url = (
        f"https://iss.moex.com/iss/engines/stock/markets/shares/boards/TQBR"
        f"/securities/{ticker}/candles.json"
    )

    while True:
        params = {
            "from": start,
            "till": end,
            "interval": 24,
            "start": cursor_start,
        }
        resp = requests.get(base_url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        columns = data["candles"]["columns"]
        rows = data["candles"]["data"]

        if not rows:
            break

        all_rows.extend(rows)
        cursor_start += len(rows)

        if len(rows) < 500:
            break

        time.sleep(0.2)

    df = pd.DataFrame(all_rows, columns=columns)
    df["Date"] = pd.to_datetime(df["begin"]).dt.date
    df = df[["Date", "close"]].rename(columns={"close": ticker})
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def get_data():
    """
    Скачиваем тренировочные и бэктестовые данные с MOEX ISS API.
    Сохраняем в train_data.parquet и backtest_data.parquet.
    """
    os.makedirs(project_path.as_posix() + "/data", exist_ok=True)
    os.makedirs(project_path.as_posix() + "/artifacts", exist_ok=True)
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    ticker_a = cfg["ticker_a"]
    ticker_b = cfg["ticker_b"]
    train_start = cfg["train_start_date"]
    train_end = cfg["train_end_date"]
    backtest_start = cfg["backtest_start_date"]
    backtest_end = cfg["backtest_end_date"]

    print(f"Downloading {ticker_a}...")
    df_a = download_moex_candles(ticker_a, train_start, backtest_end)
    print(f"  Got {len(df_a)} rows")

    print(f"Downloading {ticker_b}...")
    df_b = download_moex_candles(ticker_b, train_start, backtest_end)
    print(f"  Got {len(df_b)} rows")

    merged = pd.merge(df_a, df_b, on="Date", how="inner").sort_values("Date")
    merged = merged.dropna()
    merged = merged.astype({ticker_a: float, ticker_b: float})
    merged.index = pd.to_datetime(merged["Date"])
    merged = merged.drop(columns=["Date"]).reset_index()

    train = merged[merged["Date"] <= train_end]
    backtest = merged[merged["Date"] >= backtest_start]

    print(f"Train: {len(train)} rows ({train['Date'].min()} — {train['Date'].max()})")
    print(f"Backtest: {len(backtest)} rows ({backtest['Date'].min()} — {backtest['Date'].max()})")

    train.to_parquet(project_path.as_posix() + "/data/train_data.parquet")
    backtest.to_parquet(project_path.as_posix() + "/data/backtest_data.parquet")


if __name__ == "__main__":
    get_data()
