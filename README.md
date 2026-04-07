# pairs_trading
Статистический арбитраж: доллар-нейтральная парная торговля на акциях LKOH и ROSN на Мосбирже. Сигналы по пороговым значениям Z-счета, исполнение по открытию следующей свечи, собственный движок для бэктестинга с учётом комиссий, маржин-коллов и залогового обеспечения. Обучение модели на данных 2016–2022 годов, бэктестирование на 2023–2026 годы
## Pair Trading: LKOH / ROSN

### Структура проекта

```bash
pair_trading_project/
│
├── pair_trading_project/
│   ├── artifacts/
│   │   ├── signals_generator.joblib
│   │   ├── backtest_metrics.json
│   │   ├── requirements_plot.png
│   │   └── pnl_plot.png
│   ├── data/
│   │   ├── train_data.parquet
│   │   └── backtest_data.parquet
│   └── src/
│       ├── ArbitragePortfolio.py
│       ├── get_data.py
│       ├── run_backtest.py
│       ├── strategy.py
│       └── utils.py
├── main.py
├── config.yaml
├── pyproject.toml
├── DESCRIPTION.md
└── README.md
```

### Установка и запуск

```bash
pip install poetry==1.8.5
poetry install
python main.py
```

### Стратегия

Ratio Z-Score Mean Reversion на паре ЛУКОЙЛ / Роснефть.
Данные скачиваются с MOEX ISS API.
Подробное описание — в DESCRIPTION.md.
