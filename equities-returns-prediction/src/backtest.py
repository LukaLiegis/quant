import numpy as np

from src.metrics import compute_fees
from src.signals import compute_signal_weight
from src.config import BacktestResults, WalkForwardConfig
from src.train_models import train_models, predict_all_assets


def run_walk_forward(
        X: np.ndarray,
        Y: np.ndarray,
        symbols: list[str],
) -> BacktestResults:
    """
    Run walk forward
    """
    all_positions = []
    all_actuals = []

    fold = 1
    start_idx = 0

    while start_idx + WalkForwardConfig.train_size + WalkForwardConfig.test_size <= len(X):
        train_end = start_idx + WalkForwardConfig.train_size
        test_end = train_end + WalkForwardConfig.test_size

        X_train = X[start_idx:train_end]
        Y_train = Y[start_idx:train_end]
        X_test = X[train_end:test_end]
        Y_test = Y[train_end:test_end]

        models = train_models(X_train=X_train, Y_train=Y_train, symbols=symbols)
        predictions = predict_all_assets(models=models, X_test=X_test)
        positions = compute_signal_weight(predictions=predictions, max_positions=WalkForwardConfig.max_positions)

        all_positions.append(positions)
        all_actuals.append(Y_test)

        period_returns = (positions * Y_test).sum(axis=1)
        fold_return = (1 + period_returns).prod() - 1

        print(f'Fold {fold}: Train[{start_idx}:{train_end}] Test[{train_end}:{test_end}] Return: {fold_return:.2%}')

        start_idx += WalkForwardConfig.step_size
        fold += 1

    all_positions = np.vstack(all_positions)
    all_actuals = np.vstack(all_actuals)

    fees_per_period, total_turnover = compute_fees(positions=all_positions, fee_rate=WalkForwardConfig.fee_rate)
    total_fees = fees_per_period.sum()

    gross_returns = (all_positions * all_actuals).sum(axis=1)
    portfolio_returns = gross_returns - fees_per_period

    benchmark_returns = all_actuals.mean(axis=1)

    return BacktestResults(
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        positions=all_positions,
        actuals=all_actuals,
        symbols=symbols,
        total_fees=total_fees,
        turnover=total_turnover,
    )