import polars as pl
import pandas as pd

file_path = "../usa_00002/usa_00002.dta"

columns_to_drop = [
    'occ', 'occ2010', 'ind',
    'inctot', 'incwage', 'incbus00', 'incearn', 'poverty', 'sei', 'hesei',
    'tranwork', 'trantime', 'wkswork2', 'uhrswork'
]

print("Reading file metadata...")
iterator = pd.read_stata(file_path, iterator=True, convert_categoricals=False)
all_columns = list(iterator.variable_labels().keys())
columns_to_keep = [col for col in all_columns if col not in columns_to_drop]

print(f"Keeping {len(columns_to_keep)} of {len(all_columns)} columns")

chunksize = 500000
chunks_list = []

print("\nLoading chunks into memory...")
for i, chunk in enumerate(pd.read_stata(file_path, 
                                        chunksize=chunksize,
                                        columns=columns_to_keep, 
                                        convert_categoricals=False)):
    
    chunks_list.append(chunk)
    
    memory_mb = chunk.memory_usage(deep=True).sum() / 1e6
    print(f"Chunk {i+1}: {len(chunk):,} rows | {memory_mb:.1f} MB")

print("\nConcatenating all chunks with pandas...")
df_final = pd.concat(chunks_list, ignore_index=True)

print(f"Final shape: {df_final.shape}")
print(f"Converting to Polars and writing to parquet...")

# Convert to Polars for efficient parquet writing
pl_final = pl.from_pandas(df_final)
pl_final.write_parquet("ipums_data.parquet")

print("\n✓ Complete!")