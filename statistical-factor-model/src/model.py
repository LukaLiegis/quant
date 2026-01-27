import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from src.math import (
    winsorize,
    _compute_exponential_weights,
    _shrink_eigenvalues,
    _align_eigenvector_signs,
    _rotate_loadings
)


def _two_stage_pca(
        R: np.ndarray,
        n_factors: int,
        half_life_fast: float,
        half_life_slow: float,
        n_preliminary: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform two-stage PCA.
    """
    n_assets, T = R.shape
    gamma = n_assets / T

    W_fast = _compute_exponential_weights(T, half_life_fast)
    R_fast = R * W_fast[np.newaxis, :]

    U_1, S_1, Vt_1 = np.linalg.svd(R_fast, full_matrices=False)

    if n_preliminary > 0 and n_preliminary < min(n_assets, T):
        R_factor = U_1[:, :n_preliminary] @ np.diag(S_1[:n_preliminary]) @ Vt_1[:n_preliminary, :]
        E = R_fast - R_factor
    else:
        E = R_fast

    idio_var_proxy = np.mean(E ** 2, axis=1)
    idio_std_proxy = np.sqrt(np.maximum(idio_var_proxy, 1e-10))

    W_idio = 1.0 / idio_std_proxy

    W_slow = _compute_exponential_weights(T, half_life_slow)
    R_tilde = (W_idio[:, np.newaxis] * R) * W_slow[np.newaxis, :]

    U_2, S_2, _ = np.linalg.svd(R_tilde, full_matrices=False)

    n_factors = min(n_factors, len(S_2))
    U_N = U_2[:, :n_factors]
    S_N = S_2[:n_factors]

    eigenvalues = S_N ** 2 / T

    shrinked_eigenvalues = _shrink_eigenvalues(eigenvalues, gamma)

    min_eigenvalue = max(gamma / 10, 0.1)
    shrinked_eigenvalues = np.maximum(shrinked_eigenvalues, min_eigenvalue)

    if n_factors < len(S_2):
        remaining_eigenvalues = S_2[n_factors:] ** 2 / T
        lambda_bar = np.mean(remaining_eigenvalues)
    else:
        lambda_bar = 1.0

    lambda_bar = max(lambda_bar, 1.0)

    loadings = U_N * np.sqrt(n_assets)

    idio_variances = lambda_bar * idio_std_proxy ** 2

    return loadings, shrinked_eigenvalues, idio_variances


def _estimate_factor_returns(
        returns: np.ndarray,
        loadings: np.ndarray,
        idio_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate factor returns for single time period via cross-sectional regression.
    """
    W = np.diag(1.0 / np.maximum(idio_variances, 1e-10))

    BtWB = loadings.T @ W @ loadings
    BtWr = loadings.T @ W @ returns

    factor_returns = np.linalg.lstsq(BtWB, BtWr, rcond=None)[0]

    residual_returns = returns - loadings @ factor_returns

    return factor_returns, residual_returns


def estimate_factor_returns(
        returns_df: pl.DataFrame,
        n_factors: int = 10,
        half_life_fast: float = 21.0,
        half_life_slow: float = 126.0,
        window_size: int = 252,
        n_preliminary_factors: int = 5,
        winsor_factor: float | None = 0.05,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Estimate statistical factors.
    """
    try:
        dates_in_data = sorted(returns_df['date'].unique().to_list())
    except AttributeError as e:
        raise TypeError("`returns_df` must be a Polars DataFrame, but it's missing required attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`returns_df` must have columns 'date', 'symbol' and 'asset_returns'") from e

    try:
        symbols = sorted(returns_df['symbol'].unique().to_list())
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`returns_df` must have columns 'date', 'symbol' and 'asset_returns'") from e

    symbol_to_idx = {s: i for i, s in enumerate(symbols)}
    n_symbols = len(symbols)

    returns_wide = returns_df.pivot(
        values='asset_returns',
        index='date',
        on='symbol'
    ).sort('date')

    returns_matrix = returns_wide.select(pl.exclude('date')).to_numpy()

    if winsor_factor is not None:
        for i in range(returns_matrix.shape[0]):
            row = returns_matrix[i, :]
            valid_mask = ~np.isnan(row)
            if np.sum(valid_mask) > 0:
                returns_matrix[i, valid_mask] = winsorize(row[valid_mask], winsor_factor)

    factor_returns_list = []
    residual_returns_list = []
    loadings_list = []
    prev_loadings = None

    for t_idx, current_date in enumerate(dates_in_data):
        if t_idx < window_size - 1:
            continue

        start_idx = max(0, t_idx - window_size + 1)
        end_idx = t_idx + 1

        R_window = returns_matrix[start_idx:end_idx, :].T

        valid_assets = ~np.any(np.isnan(R_window), axis=1)
        n_valid = np.sum(valid_assets)

        if n_valid < n_factors + 1:
            continue

        R_valid = R_window[valid_assets, :]
        valid_symbols = [symbols[i] for i in range(n_symbols) if valid_assets[i]]

        try:
            loadings, shrinked_eigs, idio_vars = _two_stage_pca(
                R_valid,
                n_factors,
                half_life_fast,
                half_life_slow,
                n_preliminary_factors,
            )
        except np.linalg.LinAlgError:
            continue

        if prev_loadings is not None and prev_loadings.shape == loadings.shape:
            loadings = _align_eigenvector_signs(prev_loadings, loadings)
            loadings = _rotate_loadings(prev_loadings, loadings)

        prev_loadings = loadings.copy()

        current_returns = R_valid[:, -1]
        fac_ret, resid_ret = _estimate_factor_returns(
            current_returns,
            loadings,
            idio_vars,
        )

        factor_returns_list.append({
            'date': current_date,
            **{f'factor_{i + 1}': fac_ret[i] for i in range(len(fac_ret))},
        })

        for sym, res in zip(valid_symbols, resid_ret):
            residual_returns_list.append({
                'date': current_date,
                'symbol': sym,
                'residual_returns': res,
            })

        for sym_idx, sym in enumerate(valid_symbols):
            loadings_list.append({
                'date': current_date,
                'symbol': sym,
                **{f'factor_{i+1}': loadings[sym_idx, i] for i in range(loadings.shape[1])},
            })

        if t_idx % 1000 == 0:
            print(f'Processed {t_idx} / {len(dates_in_data)}')

    if not factor_returns_list:
        raise ValueError("Insufficient data to estimate factor model. Check window_size and min_history parameters.")

    factor_ret_df = pl.DataFrame(factor_returns_list)
    residual_ret_df = pl.DataFrame(residual_returns_list)
    loadings_df = pl.DataFrame(loadings_list)

    return factor_ret_df, residual_ret_df, loadings_df