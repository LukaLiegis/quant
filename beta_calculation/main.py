import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


def calculate_returns(prices):
    return np.log(prices / prices.shift(1)).dropna()


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
        'betas': betas,
        'alphas': alphas,
        'r_squared': r_squared,
    }, index = aligned_data.index)

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
        predicted_betas,
        realized_betas,
) -> float:
    mask = ~(np.isnan(predicted_betas) | np.isnan(realized_betas))

    if mask.sum() == 0:
        return np.nan

    pred_clean = predicted_betas[mask]
    real_clean = realized_betas[mask]

    return np.sqrt(mean_squared_error(pred_clean, real_clean))


def plot_rolling_betas(beta_df):

    ten_years_ago = pd.Timestamp.now() - pd.DateOffset(years = 10)
    recent_data = beta_df.loc[beta_df.index < ten_years_ago]

    fig, ax = plt.subplots(figsize = (16, 10))

    recent_data['OLS_beta'].plot(ax = ax, color = 'blue', linestyle = '-', label = 'OLS Beta')
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

realized_beta = calculate_realized_beta(stock_returns, market_returns)

print(basic_beta)


