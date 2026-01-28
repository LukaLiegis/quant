import pandas as pd

def get_data() -> pd.DataFrame:
    df = pd.read_csv('xnas-itch-20180501-20250430.ohlcv-1s.csv.zst', compression='zstd')

    return df

data = get_data()

print(data.head())

print(data.shape)