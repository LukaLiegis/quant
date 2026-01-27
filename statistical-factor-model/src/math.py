import numpy as np


def winsorize(
        data: np.ndarray,
        percentile: float = 0.05,
        axis: int = 0
) -> np.ndarray:
    """
    Windorize each vector of a 2D numpy array to symmetric percentiles given by `percentile`.
    """
    try:
        if not 0 <= percentile <= 1:
            raise ValueError("`percentile` must be between 0 and 1")
    except AttributeError as e:
        raise TypeError("`percentile` must be a numeric type, such as an int or float") from e

    fin_data = np.where(np.isfinite(data), data, np.nan)

    lower_bounds = np.nanpercentile(fin_data, percentile * 100, axis=axis, keepdims=True)
    upper_bounds = np.nanpercentile(fin_data, (1 - percentile) * 100, axis=axis, keepdims=True)

    return np.clip(data, lower_bounds, upper_bounds)


def _compute_exponential_weights(
        T: int,
        half_life: float,
) -> np.ndarray:
    """
    Compute exponential decay weights.
    """
    if half_life <= 0 or np.isinf(half_life):
        return  np.ones(T) / np.sqrt(T)

    t = np.arange(T, 0, -1)
    weights = np.exp(-t / half_life)

    weights = weights * np.sqrt(T / np.sum(weights ** 2))
    return weights


def _shrink_eigenvalues(
        eigenvalues: np.ndarray,
        gamma: float,
) -> np.ndarray:
    """
    Apply eigenvalue shrinkage based on spiked covariance.
    """
    threshold = 1 + np.sqrt(gamma)
    shrinked = np.where(
        eigenvalues >= threshold,
        np.maximum(eigenvalues - gamma, 0),
        eigenvalues
    )
    return shrinked


def _align_eigenvector_signs(
        U_prev: np.ndarray | None,
        U_current: np.ndarray,
) -> np.ndarray:
    """
    Align eigenvector signs to minimize turnover between consecutive periods.
    """
    if U_prev is None:
        return U_current

    n_factors = min(U_prev.shape[1], U_current.shape[1])

    cos_sims = np.einsum('ij,ij->j', U_prev[:, :n_factors], U_current[:, :n_factors])

    signs = np.ones(U_current.shape[1])
    signs[:n_factors] = np.where(cos_sims < 0, -1, 1)

    return U_current * signs


def _rotate_loadings(
        B_prev: np.ndarray | None,
        B_current: np.ndarray,
) -> np.ndarray:
    """
    Rotate loadings to minimize Frobenius distance from previous period.
    """
    if B_prev is None:
        return B_current

    n_prev = B_prev.shape[1]
    n_curr = B_current.shape[1]

    if n_prev != n_curr:
        return B_current

    A = B_prev.T @ B_current
    U, _, Vt = np.linalg.svd(A)
    Q = Vt.T @ U.T

    return B_current @ Q