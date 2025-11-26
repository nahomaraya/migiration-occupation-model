from typing import Dict, List, Optional, Tuple
import polars as pl
import gc

def create_dml_sample(
        data_path: str,
        sample_fraction: float = 0.1,
        seed: int = 42,
        save_sample: bool = False,
        output_path: Optional[str] = None,
        verbose: bool = True
) -> str:
    """
    Step 0: Create a statistically sufficient sample for DML.

    Creates a 10% sample (~3.5M rows) and saves it to disk.
    Returns path to the sampled file for use with preprocess_data_xgboost.

    Args:
        data_path: Path to full parquet/csv file
        sample_fraction: Fraction to sample (0.1 = 10%)
        seed: Random seed for reproducibility
        save_sample: Whether to save sample to disk (recommended)
        output_path: Path to save sample
        verbose: Print progress

    Returns:
        Path to sampled parquet file
    """
    if verbose:
        print("=" * 80)
        print("STEP 0: CREATE RAM-FRIENDLY SAMPLE")
        print("=" * 80)
        print(f"\n  Source: {data_path}")

    # Load with Polars (memory efficient)
    if data_path.endswith('.parquet'):
        df = pl.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pl.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")

    total_rows = len(df)

    if verbose:
        print(f"  Total rows: {total_rows:,}")
        print(f"  Sample fraction: {sample_fraction * 100:.0f}%")

    # Create sample
    df_sample = df.sample(fraction=sample_fraction, seed=seed)

    if verbose:
        print(f"  ✓ Sampled: {len(df_sample):,} rows")
        print(f"  ✓ Memory: {df_sample.estimated_size('mb'):.1f} MB")

    # Free original
    del df
    gc.collect()

    # Save sample to disk
    if output_path is None:
        output_path = data_path.replace('.parquet', '_dml_sample.parquet')

    df_sample.write_parquet(output_path)

    if verbose:
        print(f"  ✓ Saved sample to: {output_path}")
        print("\n  Note: 10% sample is statistically sufficient for DML")

    # Free sample from memory (we'll reload via preprocessor)
    del df_sample
    gc.collect()

    return output_path
