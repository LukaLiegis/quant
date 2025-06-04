import polars as pl
import matplotlib.pyplot as plt

data = pl.read_excel(
    'Commodities for the Long Run Index Level Data Monthly.xlsx',
    sheet_name='Commodities for the Long Run',
    read_options={'header_row': 10},
)

df_cumulative = data.with_columns(
    futures_return = (1 + pl.col('Excess return of equal-weight commodities portfolio').cum_sum() - 1),
    spot_return = (1 + pl.col('Excess spot return of equal-weight commodities portfolio').cum_sum() - 1),
    interest_rate_adjusted = (1 + pl.col('Interest rate adjusted carry of equal-weight commodities portfolio').cum_sum() - 1),
)

plt.figure(figsize=(12, 8))
plt.plot(df_cumulative['futures_return'], label='Excess return', color='blue')
plt.plot(df_cumulative['spot_return'], label='Spot return', color='red')
plt.plot(df_cumulative['interest_rate_adjusted'], label='Interest rate adjusted return', color='green')
plt.title('Excess Spot Return/Interest Rate Adjusted Carry Return Decomposition')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()