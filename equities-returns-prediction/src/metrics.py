import numpy as np


def compute_max_drawdown(
        returns: np.ndarray,
) -> float:
    """
    Compute maximum drawdown of a returns series.
    """
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    return drawdowns.min()


def compute_sharpe(
        returns: np.ndarray,
        periods_per_year: int = 365 * 288,
) -> float:
    """
    Compute annualized Sharpe Ratio.
    """
    std = returns.std()
    return (returns.mean() / std * np.sqrt(periods_per_year)) if std > 0 else 0.0


def compute_fees(
        positions: np.ndarray,
        fee_rate: float,
) -> tuple[np.ndarray, float]:
    """
    Compute total fees.
    """
    position_changes = np.diff(positions, axis=0, prepend=0)
    position_changes[0] = positions[0]

    turnover_per_period = np.abs(position_changes).sum(axis=1)
    total_turnover = turnover_per_period.sum()

    fees_per_period = turnover_per_period * fee_rate

    return fees_per_period, total_turnover