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