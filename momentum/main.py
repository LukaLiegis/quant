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

data = yf.download(tickers = ALL_TICKERS, period = 'max')['Close']


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
    portfolio_returns = (positions.shift(1) * returns).sum(axis = 1)
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
    daily_costs = (position_changes * cost_per_trade).sum(axis = 1)
    return daily_costs


def calculate_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def backtest_strategy(
        prices: pd.DataFrame,
        universe_dict:Dict[str, list],
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

    portfolio_returns = (actual_positions.shift(1) * returns).sum(axis = 1)

    portfolio_returns -= transaction_costs

    portfolio_returns = portfolio_returns.dropna()

    total_return = (1 + portfolio_returns).cumprod().iloc[-1] - 1
    annual_return = (1 + portfolio_returns).resample('YE').prod().mean() - 1
    annual_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    max_dd = calculate_max_drawdown(portfolio_returns)
    hit_rate = (portfolio_returns > 0).mean()

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
            'hit_rate': hit_rate,
            'avg_transaction_cost': transaction_costs.mean(),
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