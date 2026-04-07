from pair_trading_project.src.get_data import get_data
from pair_trading_project.src.run_backtest import run_backtest
from pair_trading_project.src.strategy import train_strategy


def main():
    get_data()
    train_strategy()
    run_backtest()


if __name__ == "__main__":
    main()
