import numpy as np
from xgboost import XGBRegressor
from src.config import XGBoostConfig


def train_models(
        X_train: np.ndarray,
        Y_train: np.ndarray,
        symbols: list[str],
) -> list[XGBRegressor]:
    """
    Train xgboost model, one for each symbol.
    """
    models = []
    for i, symbol in enumerate(symbols):
        model = XGBRegressor(
            n_estimators = XGBoostConfig.n_estimators,
            learning_rate = XGBoostConfig.learning_rate,
            max_depth = XGBoostConfig.max_depth,
            random_state = 42,
            n_jobs = -1,
        )
        model.fit(X_train, Y_train[:, i])
        models.append(model)
    return models


def predict_all_assets(
        models: list[XGBRegressor],
        X_test: np.ndarray,
) -> np.ndarray:
    """
    Generate predictions for all assets.
    """
    predictions = np.zeros((len(X_test), len(models)))
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(X_test)
    return predictions