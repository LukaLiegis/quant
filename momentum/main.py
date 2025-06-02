import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt


UNIVERSE = {
    'equities': ['SPY', 'QQQ', 'EFA', 'EEM', 'IWM'],
    'bonds': ['TLT', 'IEF', 'LQD', 'HYG'],
    'commodities': ['GLD', 'SLV', 'USO', 'DBA'],
    'currencies': ['UUP', 'FXE', 'FXY', 'FXA']
}

ALL_TICKERS = [ticker for sector in UNIVERSE.values() for ticker in sector]

data = yf.download(tickers = ALL_TICKERS, period = 'max')['Close']


def calculate_returns(prices):
    return np.log(prices / prices.shift(1))


def calculate_volatility_forecast(returns, short_window: int = 30, long_window: int = 252):
    short_vol = returns.rolling(short_window).std() * np.sqrt(252)
    long_vol = returns.rolling(long_window).std() * np.sqrt(252)

    vol_forecast = 0.7 * short_vol + 0.3 * long_vol

    vol_forecast = vol_forecast.ffill().bfill()

    return vol_forecast


def calculate_trend_signal(prices, fast_window: int = 60, slow_window: int = 180):
    fast_ma = prices.rolling(fast_window).mean()
    slow_ma = prices.rolling(slow_window).mean()

    returns = calculate_returns(prices)
    vol = calculate_volatility_forecast(returns)

    raw_signal = (fast_ma - slow_ma) / slow_ma
    trend_signal = raw_signal / vol

    return trend_signal.fillna(0)


def sigmoid_position_mapping(signal, steepness: int = 5, max_position: float = 1.0):
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


def calculate_position_sizes(trend_signals, vol_forecasts, weights_dict, target_vol: float = 0.15):
    raw_positions = trend_signals.apply(sigmoid_position_mapping)

    vol_adjusted_positions = raw_positions.div(vol_forecasts, axis=0)

    for ticker in vol_adjusted_positions.columns:
        if ticker in weights_dict:
            vol_adjusted_positions[ticker] *= weights_dict[ticker]

    return vol_adjusted_positions.fillna(0)


def apply_portfolio_risk_targeting(positions, returns, vol_forecast, target_vol: float = 0.15, lookback: int = 60):
    portfolio_returns = (positions.shift(1) * returns).sum(axis = 1)
    realized_vol = portfolio_returns.rolling(lookback).std() * np.sqrt(252)

    risk_scalar = target_vol / realized_vol
    risk_scalar = risk_scalar.fillna(1.0).clip(lower=0.5, upper=2.0)

    scaled_positions = positions.multiply(risk_scalar, axis=0)
    return scaled_positions


def apply_trading_buffer(current_positions, target_positions, buffer_size: float = 0.02):
    if current_positions is None:
        return target_positions

    position_diff = target_positions - current_positions
    trade_signal = np.abs(position_diff) > buffer_size

    new_positions = current_positions.copy()
    new_positions = new_positions.astype('float64')
    new_positions[trade_signal] = target_positions[trade_signal]

    return new_positions


def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def backtest_strategy(prices, universe_dict):

    returns = calculate_returns(prices)

    trend_signals = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        trend_signals[ticker] = calculate_trend_signal(prices[ticker])

    vol_forecasts = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        vol_forecasts[ticker] = calculate_volatility_forecast(returns[ticker])

    sector_weights = calculate_sector_weights(universe_dict)

    target_positions = calculate_position_sizes(trend_signals, vol_forecasts, sector_weights)

    risk_targeted_positions = apply_portfolio_risk_targeting(target_positions, returns, vol_forecasts)

    actual_positions = pd.DataFrame(index=prices.index, columns=prices.columns, dtype='float64').fillna(0.0)

    for i in range(1, len(risk_targeted_positions)):
        current_pos = actual_positions.iloc[i - 1] if i > 1 else None
        target_pos = risk_targeted_positions.iloc[i]
        actual_positions.iloc[i] = apply_trading_buffer(current_pos, target_pos)

    portfolio_returns = (actual_positions.shift(1) * returns).sum(axis = 1)

    total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    annual_return = (1 + portfolio_returns).resample('YE').prod().mean() - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    max_dd = calculate_max_drawdown(returns)

    return {
        'portfolio_returns': portfolio_returns,
        'positions': actual_positions,
        'trend_signals': trend_signals,
        'metrics': {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_dd,
        }
    }


def plot_results(results, prices):
    """Plot strategy results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    portfolio_cumulative = (1 + results['portfolio_returns']).cumprod()
    spy_returns = calculate_returns(prices['SPY'])
    spy_cumulative = (1 + spy_returns).cumprod()
    ax1.plot(portfolio_cumulative.index, portfolio_cumulative.values, label='Momentum Strategy')
    ax1.plot(spy_cumulative.index, spy_cumulative.values, label='SPY Buy & Hold')
    ax1.set_title('Strategy vs SPY')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)

    rolling_sharpe = results['portfolio_returns'].rolling(252).mean() / results['portfolio_returns'].rolling(
        252).std() * np.sqrt(252)
    ax2.plot(rolling_sharpe.index, rolling_sharpe.values)
    ax2.set_title('Rolling 1-Year Sharpe Ratio')
    ax2.set_ylabel('Sharpe Ratio')
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


results = backtest_strategy(data, UNIVERSE)

print('Strategy Performance')
for metric, value in results['metrics'].items():
    if isinstance(value, float):
        print(f"{metric.replace('_', ' ').title()}: {value:.4f}")

plot_results(results, data)