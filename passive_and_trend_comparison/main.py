import numpy as np
import polars as pl
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def get_data():
    tickers = {
        'sp500': '^GSPC',
        'russell3000': '^RUA',
        'market_index': 'VTI',
        'equal_weight_500': 'RSP',
        'momentum': 'MTUM'
    }

    data: dict = {}

    for name, ticker in tickers.items():
        df_raw = yf.download(ticker, start='2014-01-01', end='2024-12-31')['Close']

        df_polars = pl.DataFrame({
            'date': df_raw.index,
            f'close': df_raw[ticker],
        }).drop_nulls()

        data[name] = df_polars

    return data


def calculate_returns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns([
        ((pl.col('close') / pl.col('close').shift(1)) - 1).alias('daily_return'),
        ((pl.col('close') / pl.col('close').shift(21)) - 1).alias('monthly_return')
    ]).drop_nulls()


def calculate_cagr(df: pl.DataFrame) -> float:
     first_entry = df.head(1)
     last_entry = df.tail(1)

     first_value = first_entry['close'][0]
     last_value = last_entry['close'][0]

     start_dt = first_entry['date'][0]
     end_dt = last_entry['date'][0]

     num_years = (end_dt - start_dt).days / 365.25

     cagr = ((last_value / first_value) ** (1 / num_years) - 1) * 100
     return cagr


def calculate_sharpe(returns: pl.DataFrame, risk_free_rate: float = 0.02) -> float:
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)


def factor_regression():
    ...


data = get_data()

data_with_returns = {}
for name, df in data.items():
    data_with_returns[name] = calculate_returns(df)

cagrs = {}
for name, df in data_with_returns.items():
    cagr = calculate_cagr(data_with_returns[name])
    cagrs[name] = cagr
    print(f"{name:20}: {cagr:6.2f}%")

#print(data_with_returns)