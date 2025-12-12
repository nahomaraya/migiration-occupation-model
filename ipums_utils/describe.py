import polars as pl

# Quick check
df = pl.read_parquet("C:\\Users\\lenovo\\Documents\\Skool\\Data Science\\Final Project\\ipums_data_ml_ready.parquet")

if __name__ == "__main__":
    print(df.shape)
    print(df.head())
    print(df.columns)