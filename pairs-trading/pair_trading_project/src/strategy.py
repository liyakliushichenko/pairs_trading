from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sktime.transformations.base import BaseTransformer

from pair_trading_project.src.utils import load_config

project_path = Path(__file__).parent.parent


class RatioZScoreTransformer(BaseTransformer):
    """Генератор сигналов на основе Z-Score ценового отношения (ratio).

    Стратегия mean reversion: отношение LKOH/ROSN имеет свойство
    возврата к среднему. Когда Z-Score отношения превышает порог,
    открываем позицию в расчёте на возврат к среднему.

    Подход соответствует Distance/Cointegration approach из
    "The Definitive Guide to Pairs Trading" (Hudson & Thames):
    используем threshold rules на Z-Score спреда для генерации сигналов.

    Parameters
    ----------
    ticker_a : str
        Тикер первого актива.
    ticker_b : str
        Тикер второго актива.
    z_score_rolling_n : int
        Длина окна для скользящего Z-Score.
    entry_threshold : float
        Порог входа в позицию (|z| > threshold).
    exit_threshold : float
        Порог выхода из позиции (|z| < threshold).
    """

    _tags = {
        "scitype:transform-input": "Table",
        "scitype:transform-output": "Series",
        "requires_y": False,
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": False,
    }

    def __init__(
        self,
        ticker_a: str = "LKOH",
        ticker_b: str = "ROSN",
        z_score_rolling_n: int = 110,
        entry_threshold: float = 2.5,
        exit_threshold: float = 0.0,
    ):
        self.ticker_a = ticker_a
        self.ticker_b = ticker_b
        self.z_score_rolling_n = z_score_rolling_n
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self._history = None
        super().__init__()

    def _fit(self, X, y=None):
        """Сохраняет хвост обучающей выборки для инициализации
        скользящих статистик при трансформации.

        На этапе fit мы не оцениваем параметры модели (hedge ratio и т.д.),
        так как используем ratio (A/B) напрямую — оно не требует оценки
        коэффициентов. Сохраняем лишь историю для корректного
        расчёта rolling z-score в начале бэктестового периода.
        """
        self._history = X.tail(self.z_score_rolling_n).copy()
        return self

    def _transform(self, X, y=None):
        """Вычисляет Z-Score отношения цен и генерирует торговые сигналы.

        Returns
        -------
        pd.DataFrame
            Колонки: zscore, position, signal.
            signal сдвинут на 1 день вперёд (Next Open execution).
        """
        combined = pd.concat([self._history, X])
        ratio = combined[self.ticker_a] / combined[self.ticker_b]

        ratio_mean = ratio.rolling(self.z_score_rolling_n).mean()
        ratio_std = ratio.rolling(self.z_score_rolling_n).std()
        zscore = ((ratio - ratio_mean) / ratio_std).loc[X.index]

        # Генерация позиций по threshold rules
        n = len(X)
        position = np.zeros(n)
        cur = 0

        for i in range(n):
            zv = zscore.iloc[i]
            if np.isnan(zv):
                position[i] = cur
                continue

            if cur == 0:
                if zv > self.entry_threshold:
                    cur = -1  # Ratio высокий → short A, long B
                elif zv < -self.entry_threshold:
                    cur = 1   # Ratio низкий → long A, short B
            elif cur == 1 and zv > -self.exit_threshold:
                cur = 0
            elif cur == -1 and zv < self.exit_threshold:
                cur = 0

            position[i] = cur

        # Сдвигаем сигнал на 1 день (Next Open execution)
        signal = np.zeros(n)
        signal[1:] = position[:-1]

        result = pd.DataFrame(
            {"zscore": zscore.values, "position": position, "signal": signal},
            index=X.index,
        )
        return result


def train_strategy():
    """Обучаем стратегию и сохраняем обученные параметры в joblib."""
    train_data = (
        pd.read_parquet(project_path.as_posix() + "/data/train_data.parquet")
        .set_index("Date")
        .sort_index(ascending=True)
    )
    cfg = load_config(project_path.parent.as_posix() + "/config.yaml")

    transformer = RatioZScoreTransformer(
        ticker_a=cfg["ticker_a"],
        ticker_b=cfg["ticker_b"],
        z_score_rolling_n=cfg["z_score_rolling_n"],
        entry_threshold=cfg["entry_threshold"],
        exit_threshold=cfg["exit_threshold"],
    )
    transformer.fit(train_data)

    joblib.dump(
        transformer,
        project_path.as_posix() + "/artifacts/signals_generator.joblib",
    )
    print("Strategy trained and saved.")


if __name__ == "__main__":
    train_strategy()
