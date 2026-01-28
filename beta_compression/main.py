import pandas as pd
import yfinance as yf
from typing import Tuple
from matplotlib import pyplot as plt


def get_data(
        tickers,
        market_ticker: str = '^GSPC'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_tickers = tickers + [market_ticker]
    data = yf.download(all_tickers, start='2010-01-01', end='2024-12-31')['Close']
    returns = data.pct_change(fill_method=None).dropna()

    market_returns = returns[market_ticker]
    stock_returns = returns[tickers].dropna()

    return stock_returns, market_returns


def rolling_beta(
        stock_returns: pd.DataFrame,
        market_returns: pd.DataFrame,
        window: int = 63,
):
    betas = pd.DataFrame(index = stock_returns.index, columns = stock_returns.columns)

    for i, date in enumerate(stock_returns.index[window:], window):
        stock_window = stock_returns.iloc[i - window : i]
        market_window = market_returns.iloc[i - window : i]

        for ticker in stock_returns.columns:
            stock_ret = stock_window[ticker].dropna()
            aligned_market = market_window.reindex(stock_ret.index).dropna()

            common_dates = stock_ret.index.intersection(aligned_market.index)

            if len(common_dates) > 50:
                stock_aligned = stock_ret.reindex(common_dates)
                market_aligned = aligned_market.reindex(common_dates)

                covariance = stock_aligned.cov(market_aligned)
                market_variance = market_aligned.var()

                if market_variance > 0:
                    beta = covariance / market_variance
                    betas.loc[date, ticker] = beta

    return betas.astype(float)


def calculate_beta_iqr(
        betas: pd.DataFrame,
) -> pd.DataFrame:
    p75 = betas.quantile(0.75, axis=1)
    p25 = betas.quantile(0.25, axis=1)

    iqr = p75 - p25

    median_beta = betas.median(axis = 1)
    mean_beta = betas.mean(axis = 1)
    std_beta = betas.std(axis = 1)

    beta_stats = pd.DataFrame({
        'IQR': iqr,
        'P75': p75,
        'P25': p25,
        'Median': median_beta,
        'Mean': mean_beta,
        'Std': std_beta,
    })

    return beta_stats


def plot_compression(
        beta_stats: pd.DataFrame,
) -> None:

    fig, ax1 = plt.subplots(1, 1, figsize = (16, 10), dpi = 300)
    ax1.plot(beta_stats['IQR'])
    ax1.set_title('Beta Compression Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('IQR')
    ax1.grid(True)

    plt.savefig('beta_compression.png')


def main():
    tickers = [
        # Technology - High Beta
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'AMD', 'CRM',

        # Financial Services - Medium Beta
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B',

        # Consumer Defensive - Low Beta
        'WMT', 'PG', 'KO', 'PEP', 'JNJ', 'MRK', 'PFE',

        # Utilities - Very Low Beta
        'NEE', 'DUK', 'SO', 'EXC',

        # Energy - High Beta/Cyclical
        'XOM', 'CVX', 'COP', 'EOG',

        # Industrial - Medium Beta
        'BA', 'CAT', 'GE', 'MMM', 'UPS',

        # Consumer Discretionary - High Beta
        'HD', 'MCD', 'DIS', 'NKE', 'SBUX',

        # Healthcare - Medium-Low Beta
        'UNH', 'ABT', 'TMO', 'DHR',

        # Communication - Medium Beta
        'VZ', 'T', 'CMCSA',

        # REITs - Medium Beta
        'AMT', 'PLD', 'CCI'
    ]

    stock_returns, market_returns = get_data(tickers)
    betas = rolling_beta(stock_returns, market_returns)
    beta_stats = calculate_beta_iqr(betas)
    beta_stats.to_csv('beta_stats.csv')
    plot_compression(beta_stats)


if __name__ == '__main__':
    main()