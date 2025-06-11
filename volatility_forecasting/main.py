import numpy as np
import pandas as pd
import yfinance as yf


def get_data():
    data = yf.download('^GSPC', start='1986-01-01', end='2024-12-31')['Close']
    data = data.dropna()
    return data




