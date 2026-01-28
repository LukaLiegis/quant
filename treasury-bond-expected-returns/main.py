import numpy as np
import polars as pl

from src.inference import ols_with_nw
from src.plot import plot_cfc_and_yields
from src.returns import excess_returns, duration
from src.regression import vectorized_regression
from src.data_augment import construct_trend_inflation

def main():
    print("Hello from bond-project!")

    maturities = ['DGS1', 'DGS2', 'DGS5', 'DGS7', 'DGS10', 'DGS20']

    cpi_df = pl.read_csv(
        'data/CPILFENS.csv', 
        schema_overrides={'observation_date': pl.Date}
    ).fill_null(strategy="forward")

    rates_monthly = pl.read_csv(
        'data/rates_monthly.csv', 
        schema_overrides={'observation_date': pl.Date}
    )

    trend_inflation = construct_trend_inflation(cpi_df)

    combined_df = (
        trend_inflation
        .join(rates_monthly, on='observation_date', how='inner')
        .filter(pl.col('observation_date') >= pl.date(1980, 1, 1))
        .with_columns(pl.col('DGS2').cast(pl.Float64))
        .fill_null(strategy='forward')
        .drop_nulls()
    )

    cycles = vectorized_regression(combined_df, maturities)
    cycles = excess_returns(cycles)
    cycles = duration(cycles)

    # Duration-standardized average excess returns.
    cycles = cycles.with_columns([
        ((pl.col('rx_2')/2 + pl.col('rx_5')/5 + pl.col('rx_10')/10) / 3).alias('rx_bar')
    ])

    clean = cycles.select(['rx_bar', 'c_bar', 'c_DGS1']).fill_null(strategy="forward")
    X = np.column_stack([np.ones(len(clean)), clean['c_DGS1'], clean['c_bar']])
    y = clean['rx_bar'].to_numpy()
    gamma = np.linalg.lstsq(X, y, rcond=None)[0]

    cycles = cycles.with_columns(
        (gamma[0] + gamma[1] * pl.col('c_DGS1') + gamma[2] * pl.col('c_bar')).alias('cfc')
    )

    results = ols_with_nw(cycles)

    for i, ret in enumerate(results['returns']):
      beta_c1 = results['betas'][1, i]    # coefficient on c_DGS1
      beta_cbar = results['betas'][2, i]  # coefficient on c_bar
      r2 = results['r2'][i]
      print(f"{ret}: β_c1={beta_c1:.3f}, β_cbar={beta_cbar:.3f}, R²={r2:.3f}")

    plot_cfc_and_yields(cycles)

if __name__ == "__main__":
    main()