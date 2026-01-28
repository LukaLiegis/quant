import numpy as np
import polars as pl

def newey_west(
        X: np.ndarray,
        residuals: np.ndarray,
        lag: int = 18
) -> np.ndarray:
    """
    Compute Newey-West standard errors.
    """
    T, k = X.shape

    if residuals.ndim == 1:
        residuals = residuals.reshape(-1, 1)

    n_equations = residuals.shape[1]
    se = np.zeros((k, n_equations))

    for eq in range(n_equations):
        e = residuals[:, eq]

        XtX_inv = np.linalg.inv(X.T @ X)

        S = np.zeros((k, k))
        for t in range(T):
            S += np.outer(X[t] * e[t], X[t] * e[t])

        for j in range(1, lag + 1):
            weight = 1 - j / (lag + 1)
            Gamma_j = np.zeros((k, k))
            for t in range(j, T):
                Gamma_j += np.outer(X[t] * e[t], X[t - j] * e[t - j])
            S += weight * (Gamma_j + Gamma_j.T)

        var_beta = XtX_inv @ S @ XtX_inv
        se[:, eq] = np.sqrt(np.diag(var_beta))

    return se.squeeze()


def ols_with_nw(
        df: pl.DataFrame,
        returns: list[str] = ['rx_2', 'rx_5', 'rx_10'],
        predictors: list[str] = ['c_DGS1', 'c_bar'],
        lag: int = 18,
) -> dict:
    """
    OLS regression with Newey-West standard errors.
    """
    df = df.select(predictors + returns).drop_nulls()

    y = df.select(returns).to_numpy()
    X = np.column_stack([
        np.ones(len(df)),
        *[df[p].to_numpy() for p in predictors]
    ])

    betas = np.linalg.lstsq(X, y, rcond=None)[0]

    fitted = X @ betas
    residuals = y - fitted

    if y.ndim == 1:
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
    else:
        ss_res = np.sum(residuals ** 2, axis=0)
        ss_tot = np.sum((y - np.mean(y, axis=0)) ** 2, axis=0)

    r2 = 1 - ss_res / ss_tot

    se = newey_west(X, residuals, lag=lag)

    t_stats = betas / se

    return {
        'returns': returns,
        'betas': betas,
        'se': se,
        't_stats': t_stats,
        'r2': r2,
        'residuals': residuals
    }