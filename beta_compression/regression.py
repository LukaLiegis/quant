import pandas as pd
import yfinance as yf
from typing import Tuple
from matplotlib import pyplot as plt

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    beta_stats = pd.read_csv('beta_stats.csv', index_col=0, parse_dates=True)

    star_date = beta_stats.index[0] - pd.DateOffset(days=30)
    end_date = beta_stats.index[-1] + pd.DateOffset(days=100)

    spy_data = yf.download('^GSPC', start=star_date, end=end_date)
    spy_prices = spy_data['Close']['^GSPC'].dropna()
    spy_prices = pd.Series(spy_prices.values, index=spy_prices.index)

    return beta_stats, spy_prices


def main():
    beta_stats, spy_prices = load_data()


if __name__ == '__main__':
    main()