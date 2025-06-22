import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


def get_data():
    data = yf.download('^GSPC', start='1986-01-01', end='2024-12-31')['Close']
    data = data.dropna()
    return data


def realized_vol(
        prices,
        window: int = 63
) -> pd.Series:
    returns = np.log(prices / prices.shift(1)).dropna()

    rv = returns.rolling(window).std() * np.sqrt(252)

    return rv.dropna()


def create_har_features(rv_daily, returns):
    df = pd.DataFrame()
    df['rv_daily'] = rv_daily
    df['rv_weekly'] = rv_daily.rolling(window=5).mean()
    df['rv_monthly'] = rv_daily.rolling(window=22).mean()

    df['rv_daily_lag'] = df['rv_daily'].shift(1)
    df['rv_weekly_lag'] = df['rv_weekly'].shift(1)
    df['rv_monthly_lag'] = df['rv_monthly'].shift(1)

    returns_aligned = returns.loc[df.index] if len(returns) > len(df) else returns
    df['return_lag'] = returns_aligned.shift(1)

    return df.dropna()


def fit_har_model(X, y):
    X_har = X[['rv_daily_lag', 'rv_weekly_lag', 'rv_monthly_lag']]
    X_har = sm.add_constant(X_har)

    model = sm.OLS(y, X_har).fit()
    return model, X_har.columns.tolist()


def fit_kernel_ridge_model(X, y, alpha: float = 0.1, gamma: float = 0.05):
    X_kernel = X[['rv_daily_lag', 'rv_weekly_lag', 'rv_monthly_lag', 'return_lag']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_kernel)

    model = KernelRidge(alpha=alpha, gamma=gamma, kernel='rbf')
    model.fit(X_scaled, y)

    return model, scaler, X_kernel.columns.tolist()


def fit_garch_model(
        returns_train,
        returns_test,
        window_size: int = 252,
        p=1,
        q=1
):
    forecasts = []

    all_returns = pd.concat([returns_train, returns_test])

    for i in range(len(returns_test)):
        current_pos = len(returns_train) + i

        start_pos = max(0, current_pos - window_size)
        train_window = all_returns.iloc[start_pos:current_pos]

        if len(train_window) < 60:
            train_window = all_returns.iloc[:current_pos]

        train_window_pct = train_window * 100

        try:
            garch_model = arch_model(train_window_pct, vol='GARCH', p=p, q=q)
            garch_fit = garch_model.fit(disp='off')

            forecast = garch_fit.forecast(horizon=1, reindex=False)

            vol_forecasts = np.sqrt(forecast.variance.values[0, 0]) / 100 * np.sqrt(252)
            forecasts.append(vol_forecasts)

        except Exception as e:
            if forecasts:
                forecasts.append(forecasts[-1])
            else:
                vol_fallback = train_window.std() * np.sqrt(252)
                forecasts.append(vol_fallback)

            print(f'Garch convergence issue at {i + 1}, will use fallback.')
    return np.array(forecasts)


def plot_volatility_forecasts(actual, predictions):
    end_date = actual.index.max()
    start_date = end_date - pd.DateOffset(years=1)

    actual_filtered = actual[actual.index >= start_date]

    predictions_filtered = {}
    for model_name, pred_values in predictions.items():
        if isinstance(pred_values, np.ndarray):
            pred_series = pd.Series(pred_values, index=actual.index)
        else:
            pred_series = pred_values
        predictions_filtered[model_name] = pred_series[pred_series.index >= start_date]

    plt.figure(figsize=(15, 8))

    plt.plot(actual_filtered.index, actual_filtered.values, 'k-', linewidth=2, label='Actual Volatility', alpha=0.8)

    colors = ['red', 'blue', 'green']
    linestyles = ['--', '-.', ':']

    for i, (model_name, pred_values) in enumerate(predictions_filtered.items()):
        plt.plot(pred_values.index, pred_values.values,
                 color=colors[i], linestyle=linestyles[i], linewidth=2,
                 label=f'{model_name} Forecast', alpha=0.7)

    plt.title("Volatility Forecasting Comparison (Last Year)", fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Annualized Volatility', fontsize=12)
    plt.legend(fontsize=11, loc='upper right')
    plt.grid(True, alpha=0.3)

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
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
    prices = get_data()

    rv_daily = realized_vol(prices)

    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"Calculated volatility for {len(rv_daily)} observations")

    har_features = create_har_features(rv_daily, returns)

    returns_aligned = returns.loc[har_features.index]

    print(f"HAR features shape: {har_features.shape}")

    y = har_features['rv_daily'].shift(-1).dropna()
    X = har_features.iloc[:-1]
    returns_garch = returns_aligned.iloc[:-1]

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    returns_train = returns_garch.iloc[:split_idx]
    returns_test = returns_garch.iloc[split_idx:]

    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")

    results = []
    predictions = {}

    har_model, har_features_used = fit_har_model(X_train, y_train)
    print(f"HAR Model Summary:")
    print(f"R²: {har_model.rsquared:.4f}")
    print(f"Features: {har_features_used}")

    X_test_har = X_test[['rv_daily_lag', 'rv_weekly_lag', 'rv_monthly_lag']]
    X_test_har = sm.add_constant(X_test_har)
    y_pred_har = har_model.predict(X_test_har)
    predictions['HAR'] = y_pred_har

    har_results = evaluate_model(y_test, y_pred_har, 'HAR')
    results.append(har_results)

    kernel_model, scaler, kernel_features_used = fit_kernel_ridge_model(X_train, y_train)
    print(f"Kernel Ridge features: {kernel_features_used}")

    X_test_kernel = X_test[kernel_features_used]
    X_test_kernel_scaled = scaler.transform(X_test_kernel)
    y_pred_kernel = kernel_model.predict(X_test_kernel_scaled)
    predictions['Kernel Ridge'] = y_pred_kernel

    kernel_results = evaluate_model(y_test, y_pred_kernel, 'Kernel Ridge')
    results.append(kernel_results)

    print("Fitting GARCH with rolling window forecasts...")
    y_pred_garch = fit_garch_model(returns_train, returns_test, window_size=252)
    predictions['GARCH'] = y_pred_garch

    garch_results = evaluate_model(y_test, y_pred_garch, 'GARCH')
    results.append(garch_results)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, float_format='%.6f'))

    best_mse_idx = results_df['MSE'].idxmin()
    best_mse_model = results_df.loc[best_mse_idx, 'Model']
    print(f"Best MSE: {best_mse_model} ({results_df.loc[best_mse_idx, 'MSE']:.6f})")

    best_r2_idx = results_df['R²'].idxmax()
    best_r2_model = results_df.loc[best_r2_idx, 'Model']
    print(f"Best R²: {best_r2_model} ({results_df.loc[best_r2_idx, 'R²']:.6f})")

    pred_df = pd.DataFrame(predictions, index=y_test.index)
    pred_corr = pred_df.corr()
    print(f"\nCorrelation between model predictions:")
    print(pred_corr.round(4))

    print(f"\nActual volatility statistics:")
    print(f"Mean: {y_test.mean():.4f}")
    print(f"Std: {y_test.std():.4f}")
    print(f"Min: {y_test.min():.4f}")
    print(f"Max: {y_test.max():.4f}")

    print(f"\nGenerating volatility forecast comparison plot...")
    plot_volatility_forecasts(y_test, predictions)

    return results_df, predictions, y_test


if __name__ == '__main__':
    results_df, predictions, actual_volatility = main()