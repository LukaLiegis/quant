import polars as pl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_cfc_and_yields(
        df: pl.DataFrame
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    dates = df['observation_date'].to_numpy()

    ax1.plot(dates, df['DGS1'], label='1Y yield', linewidth=1)
    ax1.plot(dates, df['DGS10'], label='10Y yield', linewidth=1)
    ax1.set_ylabel('Yield (%)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)
    ax1.set_title('Treasury Yields and Risk Premium Factor', fontsize=14, fontweight='bold')

    ax2.plot(dates, df['cfc'], label='Cycle factor (cfc)', color='darkred', linewidth=1.5)
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
    ax2.set_ylabel('Risk premium', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))

    plt.savefig('plots/cfc_yields.png', dpi=300)
    plt.show()