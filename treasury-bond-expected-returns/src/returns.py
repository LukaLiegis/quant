import polars as pl

def excess_returns(
        df: pl.DataFrame
) -> pl.DataFrame:
    """
    Calculate 1-year holding period excess returns.
    """
    return df.with_columns([
        # 2-year bond
        (-1 * pl.col('DGS1').shift(-12) + 2 * pl.col('DGS2') - pl.col('DGS1')).alias('rx_2'),
        # 5-year bond
        (-4 * pl.col('DGS5').shift(-12) + 5 * pl.col('DGS5') - pl.col('DGS1')).alias('rx_5'),
        # 10-year bond
        (-9 * pl.col('DGS10').shift(-12) + 10 * pl.col('DGS10') - pl.col('DGS1')).alias('rx_10'),
    ])

def duration(
        df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Average cycle (duration standardized)
    """
    return df.with_columns([
        ((pl.col('c_DGS2') / 2 + pl.col('c_DGS5') / 5 + pl.col('c_DGS10') / 10) / 3).alias('c_bar')
    ])