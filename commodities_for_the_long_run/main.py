import polars as pl
import matplotlib.pyplot as plt

data = pl.read_excel(
    'Commodities for the Long Run Index Level Data Monthly.xlsx',
    sheet_name='Commodities for the Long Run',
    read_options={'header_row': 10},
)

data = data.rename({'__UNNAMED__0': 'date'})

data = data.with_columns(
    (1 + pl.col('Excess return of equal-weight commodities portfolio').cum_sum() - 1).alias('futures_return'),
     (1 + pl.col('Excess spot return of equal-weight commodities portfolio').cum_sum() - 1).alias('spot_return'),
    (1 + pl.col('Interest rate adjusted carry of equal-weight commodities portfolio').cum_sum() - 1).alias('interest_rate_adjusted'),
)

print(data.head())

plt.figure(figsize=(12, 8))
plt.plot(data['futures_return'], label='Excess return', color='blue')
plt.plot(data['spot_return'], label='Spot return', color='red')
plt.plot(data['interest_rate_adjusted'], label='Interest rate adjusted return', color='green')
plt.title('Excess Spot Return/Interest Rate Adjusted Carry Return Decomposition')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()