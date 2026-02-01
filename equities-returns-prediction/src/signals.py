import numpy as np

def compute_signal_weight(
        predictions: np.ndarray,
        max_position: float | None = None,
) -> np.ndarray:
    """
    Convert predictions to position weights via cross-sectional z-score.
    """
    mean = predictions.mean(axis=1, keepdims=True)
    std = predictions.std(axis=1, keepdims=True)
    std = np.where(std == 0, 1, std)

    z_scores = (predictions - mean) / std
    z_scores = z_scores / 2

    clip_limit = max_position if max_position is not None else 1.0
    positions = np.clip(z_scores, -clip_limit, clip_limit)

    return positions
