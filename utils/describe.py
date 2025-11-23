import polars as pl

# Quick check
df = pl.read_parquet("ipums_data.parquet")
print(df.shape)
print(df.head())
print(df.columns)