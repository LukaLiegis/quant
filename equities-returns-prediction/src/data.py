import numpy as np
import polars as pl

def load_and_prepare_data(
        filepath: str = 'data/combined_returns_5m.csv',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Load returns data.
    """
    returns_df = pl.read_csv(filepath).with_columns([
        pl.col('date').str.to_datetime(),
    ])

    returns_wide = returns_df.pivot(
        values='asset_returns',
        index='date',
        on='symbol'
    ).sort('date').fill_null(strategy='forward').drop_nulls()

    symbols = (returns_wide.select(pl.exclude('date')).columns).to_list()
    dates = returns_wide.select(pl.col('date')).to_numpy()
    returns_matrix = returns_wide.select(symbols).to_numpy()

    future_returns = np.roll(returns_matrix, -1, axis=0)
    future_returns[-1, :] = 0

    dates = dates[:-1]
    X = returns_matrix[:-1]
    Y = future_returns[:-1]

    return X, Y, dates, symbols