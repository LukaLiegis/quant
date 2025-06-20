import pandas as pd
import yfinance as yf


def get_data(
        tickers,
        market_ticker: str = '^GSPC'
):
    all_tickers = tickers + market_ticker
    data = yf.download(all_tickers, start='2000-01-01', end='2024-12-31', progress=False)['Close']
    returns = data.pct_change().dropna()

    market_returns = returns[market_ticker]
    stock_returns = returns[tickers].dropna()

    return stock_returns, market_returns


def rolling_beta():
    ...


def main():
    ...


if __name__ == '__main__':
    main()