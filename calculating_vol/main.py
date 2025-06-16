import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt

def get_spy_data() -> pd.Series:
    spy_df = yf.download(tickers="^GSPC", period="max")['Close']
    spy_df = spy_df.rename({'^GSPC': 'price'}, axis=1)
    spy_df['returns'] = (spy_df['price'] / spy_df['price'].shift(1)) - 1
    spy_df['log_returns'] = np.log(spy_df['price'] / spy_df['price'].shift(1))
    spy_df = spy_df.dropna()
    return spy_df


def plot(df: pd.Series) -> None:
    plt.plot(df['log_returns'], label="Log Returns")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    data = get_spy_data()
    plot(data)
    print(data)

if __name__ == "__main__":
    main()