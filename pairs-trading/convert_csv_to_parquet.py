"""
Утилита для конвертации CSV в Parquet.
"""
import pandas as pd
from pathlib import Path

data_dir = Path("pair_trading_project/data")

for name in ["train_data", "backtest_data"]:
    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"

    if csv_path.exists() and not parquet_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df.to_parquet(parquet_path)
        print(f"Converted {csv_path} -> {parquet_path}")
    elif parquet_path.exists():
        print(f"{parquet_path} already exists, skipping")
    else:
        print(f"{csv_path} not found")
