import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error


def calculate_returns(prices):
    return (prices / prices.shift(1)).dropna()


def ols_beta(
        stock_returns,
        market_returns,
        window: int = 504,
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


def ewma_beta(
        stock_returns,
        market_returns,
        window: int = 252,
        half_life: int = 63,
        min_periods: int = 126,
):
    aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
    aligned_data.columns = ['stock', 'market']

    betas = []
    alphas = []

    for i in range(len(aligned_data)):
        start_idx = max(0, i - window + 1)
        window_data = aligned_data.iloc[start_idx:i + 1]

        if len(window_data) >= min_periods:
            try:
                decay = np.log(0.5) / half_life
                weights = np.exp(decay * np.arange(len(window_data) - 1, -1, -1))
                weights = weights / weights.sum()

                y = window_data['stock'].values
                x = window_data['market'].values

                W = np.diag(weights)
                X = np.column_stack([np.ones(len(x)), x])

                coeffs = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
                alpha, beta = coeffs

                betas.append(beta)
                alphas.append(alpha)
            except:
                betas.append(np.nan)
                alphas.append(np.nan)
        else:
            betas.append(np.nan)
            alphas.append(np.nan)

    results_df = pd.DataFrame({
        'beta': betas,
        'alphas': alphas,
    }).set_index(aligned_data.index)

    return results_df


def ridge_regression_beta(
        stock_returns,
        market_returns,
        window: int = 504,
        min_periods: int = 252,
):
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    if len(aligned_data) < min_periods:
        return pd.Series(index=aligned_data.index, dtype=float)

    betas = []
    alphas = []

    for i in range(len(aligned_data)):
        start_idx = max(0, i - window + 1)
        window_data = aligned_data.iloc[start_idx:i + 1]

        if len(window_data) >= min_periods:
            try:
                y = window_data['stock'].values
                x = window_data['market'].values.reshape(-1, 1)

                ridge = Ridge(alpha = 0.5)
                ridge.fit(x, y)

                beta = ridge.coef_[0]
                alpha = ridge.intercept_

                betas.append(beta)
                alphas.append(alpha)
            except:
                betas.append(np.nan)
                alphas.append(np.nan)
        else:
            betas.append(np.nan)
            alphas.append(np.nan)

    results_df = pd.DataFrame({
        'beta': betas,
        'alphas': alphas,
    }).set_index(aligned_data.index)

    return results_df


def calculate_realized_beta(
        stock_returns,
        market_returns,
        forward_window: int = 21,
) -> pd.Series:
    aligned_data = pd.concat([stock_returns, market_returns], axis = 1).dropna()
    aligned_data.columns = ['stock', 'market']

    realized_beta = []

    for i in range(len(aligned_data) - forward_window):
        future_stock = aligned_data.iloc[i + 1: i + 1 + forward_window]['stock']
        future_market = aligned_data.iloc[i + 1: i + 1 + forward_window]['market']

        if len(future_stock) == forward_window:
            try:
                slope, _, _, _, _ = stats.linregress(future_market.values, future_stock.values)
                realized_beta.append(slope)
            except:
                realized_beta.append(np.nan)
        else:
            realized_beta.append(np.nan)

    realized_beta.extend([np.nan] * forward_window)

    return pd.Series(realized_beta, index = aligned_data.index)


def calculate_rmse(
        beta_df: pd.DataFrame,
        method_col: str,
        realized_col: str = 'Realized_Beta',
) -> float:
    clean_data = beta_df[[method_col, realized_col]].dropna()

    return np.sqrt(mean_squared_error(clean_data[realized_col], clean_data[method_col]))


def plot_rolling_betas(beta_df):

    ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years = 10)
    recent_data = beta_df.loc[beta_df.index < ten_years_ago]

    fig, ax = plt.subplots(figsize = (16, 10))

    recent_data['OLS_Beta'].plot(ax = ax, color = 'blue', linestyle = '-', label = 'OLS Beta')
    recent_data['Ridge_Regression'].plot(ax = ax, color = 'orange', linestyle = '-', label = 'Robust Regression')
    recent_data['EWMA_Short'].plot(ax=ax, color='purple', linestyle='-', label='EWMA Short')
    recent_data['EWMA_Long'].plot(ax=ax, color='brown', linestyle='-', label='EWMA Long')
    #recent_data['Realized_Beta'].plot(ax=ax, color='red', linestyle='--', label='Benchmark Beta')

    ax.set_title('2-Year Rolling Beta')
    ax.set_xlabel('Date')
    ax.set_ylabel('Beta')

    plt.legend(loc = 'best')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


stock = yf.download('AAPL')['Close']
market = yf.download('^GSPC')['Close']

stock_returns = calculate_returns(stock)
market_returns = calculate_returns(market)

basic_beta = ols_beta(stock_returns, market_returns)
ewma_short = ewma_beta(stock_returns, market_returns, half_life=63)
ewma_long = ewma_beta(stock_returns, market_returns, half_life=168)
robust_regression = ridge_regression_beta(stock_returns, market_returns)
realized_beta = calculate_realized_beta(stock_returns, market_returns)

all_beta = pd.DataFrame({
    'OLS_Beta': basic_beta['beta'],
    'EWMA_Short': ewma_short['beta'],
    'EWMA_Long': ewma_long['beta'],
    'Ridge_Regression': robust_regression['beta'],
    'Realized_Beta': realized_beta,
})

for method in ['OLS_Beta', 'Ridge_Regression', 'EWMA_Short', 'EWMA_Long']:
    rmse = calculate_rmse(all_beta, method)
    print(f"{method.replace('_', ' ')}: {rmse:.6f}")

plot_rolling_betas(all_beta)


