import polars as pl

def construct_trend_inflation(
        cpi_df: pl.DataFrame
) -> pl.DataFrame:
    """
    Docstring for construct_trend_inflation
    """
    return cpi_df.with_columns(
        (pl.col('CPILFENS').log() - pl.col('CPILFENS').shift(12).log()).alias('pi')
    ).with_columns([
        pl.col('pi').ewm_mean(alpha = 1 - 0.987, adjust=False).alias('tau_CPI')
    ]).drop_nulls()