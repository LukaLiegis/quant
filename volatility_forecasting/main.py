import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import statsmodels.api as sm
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler


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


def har_features(rv_daily, returns):
    df = pd.DataFrame()
    df['rv_daily'] = rv_daily
    df['rv_weekly'] = rv_daily.rolling(window=5).mean()
    df['rv_monthly'] = rv_daily.rolling(window=22).mean()

    df['rv_daily_lag'] = df['rv_daily'].shift(1)
    df['rv_weekly_lag'] = df['rv_weekly'].shift(1)
    df['rv_month_lag'] = df['rv_monthly'].shift(1)

    returns_aligned = returns.loc[df.index] if len(returns) > len(df) else returns
    df['return_lag'] = returns_aligned.shift(1)

    return df.dropna()


def fit_har_model(X, y):
    X_har = X[['rv_daily_lag', 'rv_weekly_lag', 'rv_month_lag']]
    X_har = sm.add_constant(X_har)

    model = sm.OLS(y, X_har).fit()
    return model, X_har.columns.tolist()


def fit_kernel_ridge_model(X, y, alpha: float = 0.1, gamma: float = 0.05):
    X_kernel = X[['rv_daily_lag', 'rv_weekly_lag', 'rv_month_lag', 'return_lag']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kernel)

    model = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
    model.fit(X_scaled, y)

    return model, scaler, X_kernel.columns.tolist()


def fit_garch_model(returns, p=1, q=1):
    returns_pct = returns * 100

    garch_model = arch_model(returns_pct, vol='GARCH', p=p, q=q)
    garch_fit = garch_model.fit(disp='off')

    return garch_fit


def plot_volatility_forecasts(actual, predictions):
    plt.figure(figsize=(15, 8))

    plt.plot(actual.index, actual.values, 'k-', linewidth=2, label='Actual Volatility', alpha=0.8)

    colors = ['red', 'blue', 'green']
    linestyles = ['--', '-.', ':']

    for i, (model_name, pred_values) in enumerate(predictions.items()):
        plt.plot(actual.index, pred_values,
                 color=colors[i], linestyle=linestyles[i], linewidth=2,
                 label=f'{model_name} Forecast', alpha=0.7)

    plt.title("Volatility Forecasting Comparison", fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    return {
        'Model': model_name,
        'MSE': mse,
        'R²': r2,
        'MAE': mae,
        'RMSE': np.sqrt(mse)
    }


def main():
    ...


if __name__ == '__main__':
    ...