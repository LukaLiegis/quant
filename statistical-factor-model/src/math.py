import numpy as np

from numpy.lib.stride_tricks import sliding_window_view


def winsorize(
        arr: np.ndarray,
        threshold: float = 5.0,
        window: int = 252,
) -> np.ndarray:
    """
    Winsorize returns using a robust z-score filter.
    """
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[np.newaxis, :]

    n_assets, n_periods = arr.shape

    if n_periods <= window:
        return arr.squeeze() if squeeze else arr.copy()

    log_ret = np.log1p(arr)
    abs_log_ret = np.abs(log_ret)

    windows = sliding_window_view(abs_log_ret, window_shape=window, axis=1)
    rolling_med = np.median(windows, axis=1)

    med_aligned = np.maximum(rolling_med[:, :-1], 1e-10)

    d = abs_log_ret[:, window:] / med_aligned

    outliers = d > threshold
    capped_abs = threshold * med_aligned
    signs = np.sign(log_ret[:, window:])
    capped_ret = np.expm1(signs * capped_abs)

    out = arr.copy()
    out[:, window:] = np.where(outliers, capped_ret, arr[:, window:])

    return out.squeeze() if squeeze else out


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