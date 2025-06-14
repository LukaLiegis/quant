import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
import matplotlib.pyplot as plt

UNIVERSE = {
    'equities': ['SPY', 'QQQ', 'EFA', 'EEM', 'IWM'],
    'bonds': ['TLT', 'IEF', 'LQD', 'HYG'],
    'commodities': ['GLD', 'SLV', 'USO', 'DBA'],
    'currencies': ['UUP', 'FXE', 'FXY', 'FXA']
}

ALL_TICKERS = [ticker for sector in UNIVERSE.values() for ticker in sector]


def download_data():
    try:
        data = yf.download(tickers=ALL_TICKERS, period='max')['Close']
        return data
    except Exception as e:
        print(f"Error downloading data: {e}")
        return None


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))


def calculate_volatility_forecast(
        returns: pd.DataFrame,
        short_window: int = 30,
        long_window: int = 252
) -> pd.DataFrame:
    short_vol = returns.rolling(short_window).std() * np.sqrt(252)
    long_vol = returns.rolling(long_window).std() * np.sqrt(252)
    vol_forecast = 0.7 * short_vol + 0.3 * long_vol
    vol_forecast = vol_forecast.ffill().bfill()
    return vol_forecast


def calculate_trend_signal(
        prices: pd.DataFrame,
        fast_window: int = 21,
        medium_window: int = 63,
        slow_window: int = 252,
) -> pd.DataFrame:
    returns = calculate_returns(prices)
    vol = calculate_volatility_forecast(returns)

    signal_1 = np.sign(returns.rolling(fast_window).sum())
    signal_2 = np.sign(returns.rolling(medium_window).sum())
    signal_3 = np.sign(returns.rolling(slow_window).sum())

    combined_signals = (
            signal_1 * 0.2 +
            signal_2 * 0.5 +
            signal_3 * 0.3
    )

    trend_signal = combined_signals / vol
    return trend_signal.fillna(0)


def sigmoid_position_mapping(
        signal: pd.DataFrame,
        steepness: float = 5,
        max_position: float = 1.0
) -> pd.DataFrame:
    return max_position * np.tanh(steepness * signal)


def calculate_sector_weights(universe_dict):
    weights_dict = {}
    n_sectors = len(universe_dict)

    for sector, tickers in universe_dict.items():
        sector_weights = 1.0 / n_sectors
        individual_weight = sector_weights / len(tickers)

        for ticker in tickers:
            weights_dict[ticker] = individual_weight

    return weights_dict


def calculate_position_sizes(
        trend_signals: pd.DataFrame,
        vol_forecasts: pd.DataFrame,
        weights_dict: Dict[str, float]
) -> pd.DataFrame:
    raw_positions = sigmoid_position_mapping(trend_signals)
    vol_adjusted_positions = raw_positions.div(vol_forecasts, axis=0)

    for ticker in vol_adjusted_positions.columns:
        if ticker in weights_dict:
            vol_adjusted_positions[ticker] *= weights_dict[ticker]

    return vol_adjusted_positions.fillna(0)


def apply_portfolio_risk_targeting(
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        target_vol: float = 0.15,
        lookback: int = 60
) -> pd.DataFrame:
    portfolio_returns = (positions.shift(1) * returns).sum(axis=1)
    realized_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(252)

    risk_scalar = target_vol / realized_vol
    risk_scalar = risk_scalar.fillna(1.0).clip(lower=0.5, upper=2.0)

    scaled_positions = positions.multiply(risk_scalar, axis=0)
    return scaled_positions


def apply_trading_buffer(
        current_positions: Optional[pd.Series],
        target_positions: pd.Series,
        buffer_size: float = 0.02
) -> pd.Series:
    if current_positions is None:
        return target_positions

    position_diff = target_positions - current_positions
    trade_signal = np.abs(position_diff) > buffer_size

    new_positions = current_positions.copy()
    new_positions = new_positions.astype('float64')
    new_positions[trade_signal] = target_positions[trade_signal]

    return new_positions


def calculate_transaction_costs(
        old_positions: pd.DataFrame,
        new_positions: pd.DataFrame,
        cost_per_trade: float = 0.001
) -> pd.Series:
    position_changes = np.abs(new_positions - old_positions)
    daily_costs = (position_changes * cost_per_trade).sum(axis=1)
    return daily_costs


def backtest_momentum_strategy(
        prices: pd.DataFrame,
        universe_dict: Dict[str, list],
) -> Dict:
    returns = calculate_returns(prices)
    trend_signals = calculate_trend_signal(prices)
    vol_forecasts = calculate_volatility_forecast(returns)
    sector_weights = calculate_sector_weights(universe_dict)
    target_positions = calculate_position_sizes(trend_signals, vol_forecasts, sector_weights)
    risk_targeted_positions = apply_portfolio_risk_targeting(target_positions, returns)

    actual_positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype='float64').fillna(0.0)
    transaction_costs = pd.Series(index=prices.index, dtype='float64').fillna(0.0)

    for i in range(1, len(risk_targeted_positions)):
        current_pos = actual_positions.iloc[i - 1] if i > 0 else None
        target_pos = risk_targeted_positions.iloc[i]

        new_pos = pd.Series(index=prices.columns, dtype='float64').fillna(0.0)
        for ticker in prices.columns:
            if not pd.isna(target_pos[ticker]):
                current_ticker_pos = current_pos[ticker] if current_pos is not None else 0
                new_pos[ticker] = apply_trading_buffer(
                    pd.Series([current_ticker_pos]),
                    pd.Series(target_pos[ticker]),
                ).iloc[0]

        actual_positions.iloc[i] = new_pos

        if i > 0:
            old_pos = actual_positions.iloc[i - 1]
            transaction_costs.iloc[i] = calculate_transaction_costs(
                old_pos.to_frame().T, new_pos.to_frame().T
            ).iloc[0]

    portfolio_returns = (actual_positions.shift(1) * returns).sum(axis=1)
    portfolio_returns -= transaction_costs
    portfolio_returns = portfolio_returns.dropna()

    return {
        'portfolio_returns': portfolio_returns,
        'positions': actual_positions,
        'returns': returns,
        'transaction_costs': transaction_costs
    }


def calculate_tilt_timing_attribution(
        positions: pd.DataFrame,
        returns: pd.DataFrame,
        horizon_days: int = 252
) -> Dict[str, pd.Series]:

    tilt_positions = positions.rolling(window=horizon_days, min_periods=60).mean()

    timing_positions = positions - tilt_positions

    actual_returns = (positions.shift(1) * returns).sum(axis=1).dropna()
    tilt_returns = (tilt_positions.shift(1) * returns).sum(axis=1).dropna()
    timing_returns = (timing_positions.shift(1) * returns).sum(axis=1).dropna()

    common_index = actual_returns.index.intersection(tilt_returns.index).intersection(timing_returns.index)

    return {
        'actual': actual_returns.loc[common_index],
        'tilt': tilt_returns.loc[common_index],
        'timing': timing_returns.loc[common_index],
        'tilt_positions': tilt_positions,
        'timing_positions': timing_positions
    }


def calculate_performance_metrics(returns: pd.Series) -> Dict[str, float]:
    total_return = (1 + returns).cumprod().iloc[-1] - 1
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_vol,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }


def plot_strategy_vs_spy(strategy_returns: pd.Series, prices: pd.DataFrame) -> None:

    spy_returns = calculate_returns(prices['SPY']).dropna()

    common_index = strategy_returns.index.intersection(spy_returns.index)
    strategy_aligned = strategy_returns.loc[common_index]
    spy_aligned = spy_returns.loc[common_index]

    strategy_cum = (1 + strategy_aligned).cumprod()
    spy_cum = (1 + spy_aligned).cumprod()

    plt.figure(figsize=(15, 8))
    plt.plot(strategy_cum.index, strategy_cum.values, label='Momentum Strategy')
    plt.plot(spy_cum.index, spy_cum.values, label='SPY Buy & Hold')
    plt.title('Strategy vs SPY')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_tilt_timing_attribution(attribution_results: Dict[str, pd.Series]) -> None:
    actual_cum = (1 + attribution_results['actual']).cumprod()
    tilt_cum = (1 + attribution_results['tilt']).cumprod()
    timing_cum = (1 + attribution_results['timing']).cumprod()

    plt.figure(figsize=(15, 8))
    plt.plot(actual_cum.index, actual_cum.values, label='Actual Strategy')
    plt.plot(tilt_cum.index, tilt_cum.values, label='Tilt Strategy')
    plt.plot(timing_cum.index, timing_cum.values, label='Timing Strategy')
    plt.title('Tilt vs Timing Attribution')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def print_attribution_summary(attribution_results: Dict[str, pd.Series]) -> None:
    strategies = ['actual', 'tilt', 'timing']
    metrics_data = []

    print(f"\n{'Strategy':<12} {'Total Ret':<10} {'Ann Ret':<10} {'Ann Vol':<10} {'Sharpe':<8} {'Max DD':<10}")
    print("-" * 70)

    for strategy in strategies:
        returns = attribution_results[strategy]
        metrics = calculate_performance_metrics(returns)
        metrics_data.append(metrics)

        print(
            f"{strategy.title():<12} {metrics['total_return']:>8.2%} {metrics['annual_return']:>8.2%} {metrics['annual_volatility']:>8.2%} {metrics['sharpe_ratio']:>6.2f} {metrics['max_drawdown']:>8.2%}")

    combined_returns = attribution_results['tilt'] + attribution_results['timing']
    correlation = np.corrcoef(attribution_results['actual'], combined_returns)[0, 1]
    mean_diff = np.abs(attribution_results['actual'] - combined_returns).mean()

    print(f"Actual vs (Tilt + Timing) correlation: {correlation:.6f}")
    print(f"Mean absolute difference: {mean_diff:.8f}")

    timing_mean = attribution_results['timing'].mean()
    print(f"Timing strategy mean return: {timing_mean:.8f}")
    print(f"(Should be close to zero: {abs(timing_mean) < 1e-6})")

    print(f"Actual Strategy Sharpe:  {metrics_data[0]['sharpe_ratio']:6.2f}")
    print(f"Tilt Strategy Sharpe:    {metrics_data[1]['sharpe_ratio']:6.2f}")
    print(f"Timing Strategy Sharpe:  {metrics_data[2]['sharpe_ratio']:6.2f}")


def main():
    data = download_data()

    if data is None:
        print("Failed to download data. Exiting.")
        return

    strategy_results = backtest_momentum_strategy(data, UNIVERSE)

    attribution_results = calculate_tilt_timing_attribution(
        strategy_results['positions'],
        strategy_results['returns'],
        horizon_days=252  # 1 year
    )

    plot_strategy_vs_spy(strategy_results['portfolio_returns'], data)
    plot_tilt_timing_attribution(attribution_results)

    print_attribution_summary(attribution_results)


if __name__ == "__main__":
    main()