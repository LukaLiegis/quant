import polars as pl

def load_yield_data() -> pl.DataFrame:
    df = pl.read_csv("combined_curves.csv")
    df = df.with_columns(
        pl.col('observation_date').str.to_date()
    ).sort("observation_date")
    return df


def main():
    df = load_yield_data()


if __name__ == "__main__":
    main()