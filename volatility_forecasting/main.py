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
    spx_data = yf.download('^GSPC', start='1986-01-01', end='2024-12-31')['Close']
    vix_data = yf.download('^VIX', start='1986-01-01', end='2024-12-31')['Close']

    common_dates = spx_data.index.intersection(vix_data.index)
    spx_data = spx_data.loc[common_dates].dropna()
    vix_data = vix_data.loc[common_dates].dropna()

    return spx_data, vix_data


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


def create_multi_horizon_targets(rv_daily, horizons):
    targets = {}
    for h in horizons:
        if h == 1:
            targets[h] = rv_daily.shift(-1)
        else:
            targets[h] = rv_daily.rolling(window=h).mean().shift(-h)

    return {k: v.dropna() for k, v in targets.items()}


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
        horizon: int = 1,
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
            forecast = garch_fit.forecast(horizon=horizon, reindex=False)

            if horizon == 1:
                vol_forecasts = np.sqrt(forecast.variance.values[0, 0]) / 100 * np.sqrt(252)
            else:
                vol_forecasts = np.sqrt(forecast.variance.values[0, :horizon].mean()) / 100 * np.sqrt(252)

            forecasts.append(vol_forecasts)

        except Exception as e:
            if forecasts:
                forecasts.append(forecasts[-1])
            else:
                vol_fallback = train_window.std() * np.sqrt(252)
                forecasts.append(vol_fallback)

    return np.array(forecasts)


def get_vix_forecasts(vix_data, test_dates):
    vix_forecasts = []

    for date in test_dates:
        try:
            vix_value = vix_data.loc[date] / 100
            vix_forecasts.append(vix_value)
        except KeyError:
            available_dates = vix_data.index[vix_data.index <= date]
            if len(available_dates) > 0:
                vix_value = vix_data.loc[available_dates[-1]] / 100
                vix_forecasts.append(vix_value)
            else:
                vix_forecasts.append(0.2)

    return np.array(vix_forecasts)


def plot_volatility_forecasts(actual, predictions):
    end_date = actual.index.max()
    start_date = end_date - pd.DateOffset(years=1)

    actual_filtered = actual[actual.index >= start_date]

    predictions_filtered = {}
    for model_name, pred_values in predictions.items():
        if isinstance(pred_values, np.ndarray):
            if len(pred_values.shape) > 1:
                pred_values = pred_values.flatten()
            pred_series = pd.Series(pred_values, index=actual.index)
        else:
            pred_series = pred_values
        predictions_filtered[model_name] = pred_series[pred_series.index >= start_date]

    plt.figure(figsize=(15, 8))

    plt.plot(actual_filtered.index, actual_filtered.values, 'k-', linewidth=2, label='Actual Volatility', alpha=0.8)

    colors = ['red', 'blue', 'green', 'orange']
    linestyles = ['--', '-.', ':', '-']

    for i, (model_name, pred_values) in enumerate(predictions_filtered.items()):
        plt.plot(pred_values.index, pred_values.values,
                 color=colors[i % len(colors)], linestyle=linestyles[i % len(linestyles)], linewidth=2,
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
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.values.flatten()
    elif isinstance(y_true, pd.Series):
        y_true = y_true.values
    elif not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)

    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.values.flatten()
    elif isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    elif not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    if len(y_pred.shape) > 1:
        y_pred = y_pred.flatten()

    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = np.mean(np.abs(y_true - y_pred))

    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, np.finfo(float).eps))) * 100

    return {
        'Model': model_name,
        'MSE': mse,
        'R²': r2,
        'MAE': mae,
        'RMSE': np.sqrt(mse),
        'MAPE': mape,
    }


def plot_metrics_vs_horizon(results_by_horizon, horizons):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    fig.suptitle('Model Performance Across Forecast Horizons', fontsize=16, fontweight='bold')

    metrics = ['MSE', 'MAPE', 'R²', 'MAE']
    metric_titles = ['Mean Squared Error', 'Mean Absolute Percentage Error (%)', 'R-Squared', 'Mean Absolute Error']

    for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
        ax = axes[idx]

        all_models = set()
        for h in horizons:
            for result in results_by_horizon[h]:
                all_models.add(result['Model'])

        colors = ['red', 'blue', 'green', 'orange', 'purple']

        for i, model in enumerate(sorted(all_models)):
            model_values = []
            for h in horizons:
                model_result = next((r for r in results_by_horizon[h] if r['Model'] == model), None)
                if model_result:
                    model_values.append(model_result[metric])
                else:
                    model_values.append(np.nan)

            ax.plot(horizons, model_values, marker='o', linewidth=2,
                    label=model, color=colors[i % len(colors)], markersize=6)

        ax.set_xlabel('Forecast Horizon (days)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xticks(horizons)

    plt.tight_layout()
    plt.show()


def main():
    prices, vix_data = get_data()

    rv_daily = realized_vol(prices)
    returns = np.log(prices / prices.shift(1)).dropna()

    print(f"Calculated volatility for {len(rv_daily)} observations")

    har_features = create_har_features(rv_daily, returns)
    returns_aligned = returns.loc[har_features.index]

    horizons = [1, 5, 30, 60]

    targets = create_multi_horizon_targets(rv_daily, horizons)

    common_index = har_features.index
    for h in horizons:
        common_index = common_index.intersection(targets[h].index)

    X = har_features.loc[common_index]
    returns_garch = returns_aligned.loc[common_index]

    max_horizon = max(horizons)
    X = X.iloc[:-max_horizon]
    returns_garch = returns_garch.iloc[:-max_horizon]

    split_idx = int(0.8 * len(X))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    returns_train = returns_garch.iloc[:split_idx]
    returns_test = returns_garch.iloc[split_idx:]

    print(f"\nTrain set: {len(X_train)} observations")
    print(f"Test set: {len(X_test)} observations")

    results_by_horizon = {}

    for horizon in horizons:
        print(f"\nForecasting {horizon}-day horizon...")

        y = targets[horizon].loc[common_index].iloc[:-max_horizon]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        if isinstance(y_test, pd.Series):
            y_test_values = y_test.values
        elif isinstance(y_test, pd.DataFrame):
            y_test_values = y_test.values.flatten()
        else:
            y_test_values = np.array(y_test)

        results = []
        predictions = {}

        har_model, _ = fit_har_model(X_train, y_train)
        X_test_har = X_test[['rv_daily_lag', 'rv_weekly_lag', 'rv_monthly_lag']]
        X_test_har = sm.add_constant(X_test_har)
        y_pred_har = har_model.predict(X_test_har)

        if isinstance(y_pred_har, pd.Series):
            y_pred_har = y_pred_har.values
        elif isinstance(y_pred_har, pd.DataFrame):
            y_pred_har = y_pred_har.values.flatten()
        if len(y_pred_har.shape) > 1:
            y_pred_har = y_pred_har.flatten()

        predictions['HAR'] = y_pred_har
        har_results = evaluate_model(y_test_values, y_pred_har, 'HAR')
        results.append(har_results)

        kernel_model, scaler, kernel_features = fit_kernel_ridge_model(X_train, y_train)
        X_test_kernel = X_test[kernel_features]
        X_test_kernel_scaled = scaler.transform(X_test_kernel)
        y_pred_kernel = kernel_model.predict(X_test_kernel_scaled)

        if isinstance(y_pred_kernel, pd.Series):
            y_pred_kernel = y_pred_kernel.values
        elif isinstance(y_pred_kernel, pd.DataFrame):
            y_pred_kernel = y_pred_kernel.values.flatten()
        if len(y_pred_kernel.shape) > 1:
            y_pred_kernel = y_pred_kernel.flatten()

        predictions['Kernel Ridge'] = y_pred_kernel
        kernel_results = evaluate_model(y_test_values, y_pred_kernel, 'Kernel Ridge')
        results.append(kernel_results)

        y_pred_garch = fit_garch_model(returns_train, returns_test, horizon=horizon, window_size=252)

        if isinstance(y_pred_garch, pd.Series):
            y_pred_garch = y_pred_garch.values
        elif isinstance(y_pred_garch, pd.DataFrame):
            y_pred_garch = y_pred_garch.values.flatten()
        if len(y_pred_garch.shape) > 1:
            y_pred_garch = y_pred_garch.flatten()

        predictions['GARCH'] = y_pred_garch
        garch_results = evaluate_model(y_test_values, y_pred_garch, 'GARCH')
        results.append(garch_results)

        vix_forecasts = get_vix_forecasts(vix_data, y_test.index)

        if isinstance(vix_forecasts, pd.Series):
            vix_forecasts = vix_forecasts.values
        elif isinstance(vix_forecasts, pd.DataFrame):
            vix_forecasts = vix_forecasts.values.flatten()
        if len(vix_forecasts.shape) > 1:
            vix_forecasts = vix_forecasts.flatten()

        predictions['VIX'] = vix_forecasts
        vix_results = evaluate_model(y_test_values, vix_forecasts, 'VIX')
        results.append(vix_results)

        results_by_horizon[horizon] = results

        results_df = pd.DataFrame(results)
        print(f"\n{horizon}-day forecast results:")
        print(results_df[['Model', 'MSE', 'R²', 'MAE', 'MAPE']].to_string(index=False, float_format='%.4f'))

    print(f"\nGenerating volatility forecast comparison plot...")
    plot_volatility_forecasts(y_test, predictions)

    plot_metrics_vs_horizon(results_by_horizon, horizons)

    return results_by_horizon


if __name__ == '__main__':
    results_by_horizon = main()