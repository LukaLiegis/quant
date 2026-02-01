import numpy as np

def compute_signal_weight(
        predictions: np.ndarray,
) -> np.ndarray:
    """
    Convert predictions to position weights via cross-sectional z-score.
    """
    mean = predictions.mean(axis=1, keepdims=True)
    std = predictions.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)

    z_scores = (predictions - mean) / std
    positions = np.clip(z_scores / 2, -1, 1)
    return positions
