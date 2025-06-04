import polars as pl

data = pl.read_excel(
    'Commodities for the Long Run Original Paper Data.xlsx',
    sheet_name='Data',
    read_options={'header_row': 10},
)

print(data.shape)