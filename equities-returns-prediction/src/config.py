import numpy as np
from dataclasses import dataclass

@dataclass
class WalkForwardConfig:
    train_size: int = 50_000
    test_size: int = 5_000
    step_size: int = 5_000

@dataclass
class XGBoostConfig:
    n_estimators: int = 200
    learning_rate: float = 0.01
    max_depth: int = 5

@dataclass
class BacktestResults:
    portfolio_returns: np.ndarray
    benchmark_returns: np.ndarray
    positions: np.ndarray
    actuals: np.ndarray
    symbols: list[str]