import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

pd.set_option('display.max_columns', None)


def get_spy_data() -> pd.DataFrame:
    spy_df = yf.download(tickers="SPY", start="1990-01-01", end="2024-12-31")
    spy_df['returns'] = (spy_df['Close'] / spy_df['Close'].shift(1)) - 1
    spy_df['log_returns'] = np.log(spy_df['Close'] / spy_df['Close'].shift(1))
    spy_df = spy_df.dropna()
    return spy_df

def get_vix_data() -> pd.Series:
    vix_df = yf.download(tickers="^VIX", start="1990-01-01", end="2024-12-31")['Close']
    vix_df = vix_df.rename({'^VIX': 'vix'}, axis=1)
    return vix_df


def close_to_close(log_returns: pd.Series, window: int = 21) -> pd.Series:
    return log_returns.rolling(window = window).std(ddof=1) * np.sqrt(252) * 100


def exponentially_weighted_volatility(log_returns: pd.Series, lambda_param: int = 0.94) -> pd.Series:
    squared_returns = log_returns ** 2
    ewma_variance = squared_returns.ewm(alpha = 1 - lambda_param, adjust = False).mean()
    volatility = np.sqrt(ewma_variance * 252) * 100
    return volatility.reindex(log_returns.index)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> pd.Series:
    hl_ratio_squared = (np.log(high / low) ** 2)
    rolling_variance = hl_ratio_squared.rolling(window = window).mean() / (4 * np.log(2))
    return np.sqrt(rolling_variance * 252) * 100


def garman_klass_volatility(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21
) -> pd.Series:
    hl_component = 0.5 * (np.log(high / low) ** 2)
    oc_component = (2 * np.log(2) - 1) * (np.log(close / open) ** 2)
    gk_component = hl_component - oc_component
    rolling_variance = gk_component.rolling(window = window).mean()
    return np.sqrt(rolling_variance * 252) * 100


def rogers_satchell_volatility(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21
) -> pd.Series:
    high_component = np.log(high / close) * np.log(high / open)
    low_component = np.log(low / close) * np.log(low / open)
    rs_component = high_component + low_component
    rolling_variance = rs_component.rolling(window = window).mean()
    return np.sqrt(rolling_variance * 252) * 100


def yang_zhang_volatility(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 21
) -> pd.Series:
    overnight = (np.log(open / close.shift(1)) ** 2)

    high_component = np.log(high / close) * np.log(high / open)
    low_component = np.log(low / close) * np.log(low / open)
    rs_component = high_component + low_component

    oc_component = (np.log(close / open) ** 2)

    yz_component = overnight + 0.5 * rs_component+ 0.5 * oc_component
    rolling_variance = yz_component.rolling(window = window).mean()
    return np.sqrt(rolling_variance * 252) * 100


def calculate_measures(data: pd.DataFrame, vix_data: pd.Series) -> pd.DataFrame:
    if isinstance(vix_data, pd.DataFrame):
        vix_data = vix_data.iloc[:, 0]

    vix_aligned = vix_data.reindex(data.index, method='ffill')

    volatility_measures = {
        'Close-to-Close': 'close_to_close_vol',
        'EWMA': 'ewma_volatility',
        'Parkinson': 'parkinson_vol',
        'Garman-Klass': 'garman_klass_vol',
        'Rogers-Satchell': 'rogers_satchell_vol',
        'Yang-Zhang': 'yang_zhang_vol'
    }

    results = []

    for measure_name, vol_col in volatility_measures.items():
        if vol_col not in data.columns:
            continue

        mask = ~(np.isnan(data[vol_col]) | np.isnan(vix_aligned))

        if mask.sum() == 0:
            continue

        predicted = data[vol_col][mask].values
        actual = vix_aligned[mask].values

        mse = mean_squared_error(actual, predicted)
        rmse = mean_squared_error(actual, predicted)
        mape = mean_absolute_percentage_error(actual, predicted) * 100
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)

        correlation, p_value = pearsonr(actual, predicted)

        results.append({
            'Volatility_Measure': measure_name,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape,
            'MAE': mae,
            'R²': r2,
            'Correlation': correlation,
            'P_Value': p_value,
            'N_Observations': mask.sum()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values('RMSE')

    return results_df


def plot(df: pd.DataFrame, vix: pd.Series) -> None:
    end_date = df.index.max()
    start_date = end_date - pd.DateOffset(years=1)
    df_filtered = df[df.index >= start_date]
    vix_filtered = vix[vix.index >= start_date]

    plt.figure(figsize=(15, 8))
    plt.plot(vix_filtered, label = "VIX")
    plt.plot(df_filtered['close_to_close_vol'], label="Close to Close Volatility")
    plt.plot(df_filtered['ewma_volatility'], label="Exponentially Weighted Volatility")
    plt.plot(df_filtered['parkinson_vol'], label="Parkinson Volatility")
    plt.plot(df_filtered['garman_klass_vol'], label="Garman-Klass Volatility")
    plt.plot(df_filtered['rogers_satchell_vol'], label="Rogers-Satchell Volatility")
    plt.plot(df_filtered['yang_zhang_vol'], label="Yang-Zhang Volatility")
    plt.title("SPY Volatility Calculations Compared to VIX")
    plt.ylabel("Volatility (%)")
    plt.xlabel("Date")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_scatter(ewma_vol: pd.Series, vix_data: pd.Series) -> None:
    if isinstance(vix_data, pd.DataFrame):
        vix_data = vix_data.iloc[:, 0]

    vix_aligned = vix_data.reindex(ewma_vol.index, method='ffill')
    mask = ~(np.isnan(ewma_vol) | np.isnan(vix_aligned))

    x_data = ewma_vol[mask].values
    y_data = vix_aligned[mask].values

    plt.figure(figsize=(10, 8))
    plt.scatter(x_data, y_data, alpha=0.6, s=20, color='blue', label='Data Points')
    plt.xlabel('EWMA Volatility (%)')
    plt.ylabel('VIX (%)')
    plt.title('EWMA Volatility vs VIX Scatter Plot')
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    data = get_spy_data()
    vix_data = get_vix_data()
    data['close_to_close_vol'] = close_to_close(data['log_returns'])
    data['ewma_volatility'] = exponentially_weighted_volatility(data['log_returns'])
    data['parkinson_vol'] = parkinson_volatility(data['High'], data['Low'])
    data['garman_klass_vol'] = garman_klass_volatility(data['Open'], data['High'], data['Low'], data['Close'])
    data['rogers_satchell_vol'] = rogers_satchell_volatility(data['Open'], data['High'], data['Low'], data['Close'])
    data['yang_zhang_vol'] = yang_zhang_volatility(data['Open'], data['High'], data['Low'], data['Close'])
    plot(data, vix_data)
    plot_scatter(data['ewma_volatility'], vix_data)

    performance_metrics = calculate_measures(data, vix_data)

    print(performance_metrics.round(2))


if __name__ == "__main__":
    main()