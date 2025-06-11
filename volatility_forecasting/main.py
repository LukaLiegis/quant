import numpy as np
import pandas as pd
import yfinance as yf


def get_data():
    data = yf.download('^GSPC', start='1986-01-01', end='2024-12-31')['Close']
    data = data.dropna()
    return data


def realized_vol(prices, window: int):
    returns = np.log(prices / prices.shift(1)).dropna()

    if window == 1:
        rv = np.abs(returns) * np.sqrt(252)
    else:
        rv = returns.rolling(window).std()* np.sqrt(252)

    return rv.dropna()




