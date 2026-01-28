import numpy as np
import polars as pl

def vectorized_regression(
        df: pl.DataFrame,
        maturities: list[str]
) -> pl.DataFrame:
    """
    Extract cycles
    """
    yields_matrix = df.select(maturities).to_numpy().astype(np.float64)
    tau = df.select('tau_CPI').to_numpy().astype(np.float64)

    X = np.column_stack([np.ones(len(tau)), tau])

    coeffs = np.linalg.lstsq(X, yields_matrix, rcond=None)[0]

    fitted = X @ coeffs

    cycles = yields_matrix - fitted

    for i, mat in enumerate(maturities):
        df = df.with_columns(
            pl.lit(cycles[:, i]).alias(f'c_{mat}')
        )

    return df