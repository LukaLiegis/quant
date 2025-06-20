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


def rolling_beta(
        stock_returns,
        market_returns,
        window: int = 252,
):
    betas = pd.DataFrame(index = stock_returns.index, columns = stock_returns.columns)

    for i, date in enumerate(stock_returns.index[window:], window):
        stock_window = stock_returns.iloc[i - window : i]
        market_window = market_returns.iloc[i - window : i]

        for ticker in stock_returns.columns:
            stock_ret = stock_window[ticker].dropna()
            aligned_market = market_window.reindex(stock_ret.index)

            valid_data = pd.concat([stock_window, aligned_market], axis=1).dropna()

            if len(valid_data) > 50:
                covariance_matrix = valid_data.cov()
                if len(covariance_matrix) > 1:
                    covariance = covariance_matrix.iloc[0, 1]
                    market_variance = covariance_matrix.iloc[1, 1]

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


def main():
    ...


if __name__ == '__main__':
    main()