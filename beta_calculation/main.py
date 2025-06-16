import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from sklearn.metrics import mean_squared_error


def calculate_log_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


def ols_beta(
        stock_returns,
        market_returns,
        window: int = 252,
        min_periods: int = 252,
) -> pd.DataFrame:
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    betas = []
    alphas = []
    r_squared = []

    for i in range(len(aligned_data)):
        start_idx = max(0, i - window + 1)
        window_data = aligned_data.iloc[start_idx:i + 1]

        if len(window_data) >= min_periods:
            y = window_data['stock'].values
            x = window_data['market'].values

            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            betas.append(slope)
            alphas.append(intercept)
            r_squared.append(r_value ** 2)
        else:
            betas.append(np.nan)
            alphas.append(np.nan)
            r_squared.append(np.nan)

    results_df = pd.DataFrame({
        'beta': betas,
        'alphas': alphas,
        'r_squared': r_squared,
    }, index = aligned_data.index)

    return results_df


def calculate_realized_beta(
        stock_returns,
        market_returns,
        forward_window: int = 126,
) -> pd.Series:
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    realized_beta = []

    for i in range(len(aligned_data)):

        if i + forward_window <= len(aligned_data):
            future_data = aligned_data.iloc[i:i + forward_window]

            if len(future_data) >= forward_window / 2:
                stock_ret = future_data['stock'].values
                market_ret = future_data['market'].values

                numerator = np.sum(stock_ret * market_ret)
                denominator = np.sum(market_ret ** 2)

                if denominator != 0:
                    beta = numerator / denominator
                    realized_beta.append(beta)
                else:
                    realized_beta.append(np.nan)
            else:
                realized_beta.append(np.nan)
        else:
            realized_beta.append(np.nan)

    return pd.Series(realized_beta, index = aligned_data.index)


def kalman_filter_beta(
        stock_returns,
        market_returns,
        process_variance: float = 1e-2,
        observation_variance: float = 1e-2,
) -> pd.Series:
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    kf = KalmanFilter(dim_x=1, dim_z=1)

    kf.F = np.array([[1.0]])
    kf.H = np.array([[1.0]])

    kf.Q = np.array([[process_variance]])
    kf.R = np.array([[observation_variance]])

    kf.x = np.array([[1.0]])
    kf.P = np.array([[1.0]])

    betas = []

    for i in range(len(aligned_data)):
        market_ret = aligned_data['market'].iloc[i]
        stock_ret = aligned_data['stock'].iloc[i]

        if not pd.isna(market_ret) and not pd.isna(stock_ret) and market_ret != 0:

            kf.H = np.array([[market_ret]])

            kf.predict()
            kf.update(stock_ret)

            betas.append(kf.x[0, 0])
        else:
            kf.predict()
            betas.append(kf.x[0, 0])

    return pd.Series(betas, index = aligned_data.index)


def ewma_beta(
        stock_returns,
        market_returns,
        decay_factor: float = 0.98,
        min_periods: int = 30
) -> pd.Series:
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    betas = []
    for i in range(len(aligned_data)):
        if i < min_periods:
            betas.append(np.nan)
            continue

        hist_data = aligned_data.iloc[:i + 1]

        n = len(hist_data)
        weights = np.array([(decay_factor ** (n - 1 - j)) for j in range(n)])
        weights = weights / weights.sum()

        stock_ret = hist_data['stock'].values
        market_ret = hist_data['market'].values

        weighted_cov = np.sum(weights * stock_ret * market_ret)
        weighted_var = np.sum(weights * market_ret ** 2)

        if weighted_var > 1e-8:
            beta = weighted_cov / weighted_var
            betas.append(beta)
        else:
            betas.append(np.nan)

    return pd.Series(betas, index = aligned_data.index)


def calculate_rmse(
        beta_df: pd.DataFrame,
        method_col: str,
        realized_col: str = 'Realized_Beta',
) -> float:
    clean_data = beta_df[[method_col, realized_col]].dropna()

    if len(clean_data) == 0:
        return np.nan

    return np.sqrt(mean_squared_error(clean_data[realized_col], clean_data[method_col]))


def plot_rolling_betas(beta_df, plot_years: int = 5):
    methods_to_plot = ['OLS_Beta', 'Kalman_Beta', 'EWMA_Beta', 'Realized_Beta']

    ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years = plot_years)
    recent_data = beta_df.loc[beta_df.index >= ten_years_ago]

    fig, ax = plt.subplots(figsize = (16, 10))

    colors = ['blue', 'green', 'orange', 'red']
    linestyles = ['-', '-', '-', '--']

    for i, method in enumerate(methods_to_plot):
        if method in recent_data.columns:
            recent_data[method].plot(
                ax=ax,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                label=method.replace('_', ' '),
                alpha=0.8
            )

    ax.set_title(f'Beta Estimates Comparison (Last {plot_years} Years)', fontsize=14)
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    stock = yf.download('AAPL')['Close']
    market = yf.download('^GSPC')['Close']

    log_stock_returns = calculate_log_returns(stock)
    log_market_returns = calculate_log_returns(market)

    basic_beta = ols_beta(log_stock_returns, log_market_returns)
    kalman_beta = kalman_filter_beta(log_stock_returns, log_market_returns)
    ewma_beta_est = ewma_beta(log_stock_returns, log_market_returns)
    realized_beta = calculate_realized_beta(log_stock_returns, log_market_returns)

    all_beta = pd.DataFrame({
        'OLS_Beta': basic_beta['beta'],
        'Kalman_Beta': kalman_beta,
        'EWMA_Beta': ewma_beta_est,
        'Realized_Beta': realized_beta,
    })

    for method in ['OLS_Beta', 'Kalman_Beta', 'EWMA_Beta', 'Realized_Beta']:
        rmse = calculate_rmse(all_beta, method)
        print(f"{method.replace('_', ' ')}: {rmse:.6f}")

    plot_rolling_betas(all_beta)


