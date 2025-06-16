import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import pyplot as plt


def get_spy_data() -> pd.DataFrame:
    spy_df = yf.download(tickers="^GSPC", start="1990-01-01", end="2024-12-31")
    spy_df['returns'] = (spy_df['Close'] / spy_df['Close'].shift(1)) - 1
    spy_df['log_returns'] = np.log(spy_df['Close'] / spy_df['Close'].shift(1))
    spy_df = spy_df.dropna()
    return spy_df

def get_vix_data() -> pd.Series:
    vix_df = yf.download(tickers="^VIX", start="1990-01-01", end="2024-12-31")['Close']
    vix_df = vix_df.rename({'^VIX': 'vix'}, axis=1)
    return vix_df


def close_to_close(log_returns: pd.Series, window: int = 30) -> pd.Series:
    return log_returns.rolling(window = window).std(ddof=1) * np.sqrt(252) * 100


def exponentially_weighted_volatility(log_returns: pd.Series, lambda_param: int = 0.94) -> pd.Series:
    squared_returns = log_returns ** 2
    ewma_variance = squared_returns.ewm(alpha = 1 - lambda_param, adjust = False).mean()
    volatility = np.sqrt(ewma_variance * 252) * 100
    return volatility.reindex(log_returns.index)


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 30) -> pd.Series:
    hl_ratio_squared = (np.log(high / low) ** 2)
    rolling_variance = hl_ratio_squared.rolling(window = window).mean() / (4 * np.log(2))
    return np.sqrt(rolling_variance * 252) * 100


def garman_klass_volatility(
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        window: int = 30
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
        window: int = 30
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
        window: int = 30
) -> pd.Series:
    overnight = (np.log(open / close.shift(1)) ** 2)

    high_component = np.log(high / close) * np.log(high / open)
    low_component = np.log(low / close) * np.log(low / open)
    rs_component = high_component + low_component

    oc_component = (np.log(close / open) ** 2)

    yz_component = overnight + 0.5 * rs_component+ 0.5 * oc_component
    rolling_variance = yz_component.rolling(window = window).mean()
    return np.sqrt(rolling_variance * 252) * 100



def calculate_measures(data: pd.DataFrame, vix_data: pd.Series):
    vix_aligned = vix_data.reindex(data.index)
    premiums = pd.DataFrame(index=data.index)

    premiums['close_to_close_premium'] = data['close_to_close_vol'] - vix_aligned
    premiums['ewma_premium'] = data['ewma_volatility'] - vix_aligned
    premiums['parkinson_premium'] = data['parkinson_vol'] - vix_aligned
    premiums['garman_klass_premium'] = data['garman_klass_vol'] - vix_aligned
    premiums['rogers_satchell_premium'] = data['rogers_satchell_vol'] - vix_aligned
    premiums['yang_zhang_premium'] = data['yang_zhang_vol'] - vix_aligned
    premiums['vix'] = vix_aligned

    premium_cols = [col for col in premiums.columns if 'premium' in col]

    summary_stats = pd.DataFrame({
        'mean': premiums[premium_cols].mean(),
        'std': premiums[premium_cols].std(),
    }).round(2)

    print(summary_stats)

    return premiums


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
    #print(data)
    #print(vix_data)


if __name__ == "__main__":
    main()