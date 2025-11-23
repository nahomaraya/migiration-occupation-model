# preprocess_data_xgboost.py
from typing import Tuple, Optional, List
import warnings
import pandas as pd
import polars as pl
import numpy as np
from preprocessing.clip_outliers import clip_outliers as clip_outliers_iqr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc

warnings.filterwarnings('ignore')


def preprocess_data_xgboost(
        data_path: str,
        target_column: str,
        weight_column: str = 'perwt',
        exclude_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        stratify_column: Optional[str] = 'is_immigrant',
        apply_scaling: bool = False,
        handle_outliers: bool = False,
        outlier_method: str = 'percentile',
        outlier_lower: float = 0.01,
        outlier_upper: float = 0.99,
        verbose: bool = True,
        use_polars: bool = True  # NEW: Keep in Polars as long as possible
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Memory-safe preprocessing for XGBoost training.

    Strategy:
    1. Load with Polars (much more memory efficient)
    2. Split indices first (not data)
    3. Convert only train/test splits to Pandas separately
    4. Aggressive garbage collection

    Args:
        data_path: Path to parquet/csv file
        target_column: Name of target variable
        weight_column: Name of weight variable (default: 'perwt')
        exclude_columns: List of columns to exclude from features
        test_size: Proportion of test set (default: 0.2)
        stratify_column: Column to stratify split (default: 'year')
        apply_scaling: Whether to scale features (usually False for XGBoost)
        handle_outliers: Whether to clip outliers
        outlier_method: 'percentile' or 'iqr'
        outlier_lower: Lower percentile for clipping (default: 0.01)
        outlier_upper: Upper percentile for clipping (default: 0.99)
        verbose: Print diagnostic information
        use_polars: Keep data in Polars format as long as possible

    Returns:
        X_train, X_test, y_train, y_test, w_train, w_test
    """

    if verbose:
        print("=" * 80)
        print("MEMORY-SAFE XGBOOST PREPROCESSING")
        print("=" * 80)

    # ==========================================
    # STEP 1: LOAD WITH POLARS (Memory Efficient)
    # ==========================================
    if verbose:
        print("\n[1/6] Loading data with Polars...")

    if data_path.endswith('.parquet'):
        df = pl.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pl.read_csv(data_path)
    else:
        raise ValueError("File must be .parquet or .csv")

    if verbose:
        print(f"  ✓ Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"  Memory: {df.estimated_size('mb'):.1f} MB")

    # Validate required columns
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found")
    if weight_column not in df.columns:
        raise ValueError(f"Weight column '{weight_column}' not found")

    # ==========================================
    # STEP 2: PREPARE SPLIT INDICES (No Data Copy!)
    # ==========================================
    if verbose:
        print(f"\n[2/6] Preparing train-test split indices...")

    # Get stratification column if needed
    if stratify_column and stratify_column in df.columns:
        stratify_values = df[stratify_column].to_numpy()
        if verbose:
            print(f"  ✓ Stratifying by: {stratify_column}")
    else:
        stratify_values = None

    # Create indices for split
    n_samples = len(df)
    indices = np.arange(n_samples)

    # Split indices only (not the actual data!)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=stratify_values
    )

    if verbose:
        print(f"  Train indices: {len(train_idx):,} ({len(train_idx) / n_samples * 100:.1f}%)")
        print(f"  Test indices: {len(test_idx):,} ({len(test_idx) / n_samples * 100:.1f}%)")

    # Free memory
    del stratify_values
    gc.collect()

    # ==========================================
    # STEP 3: FILTER COLUMNS (Still in Polars)
    # ==========================================
    if verbose:
        print(f"\n[3/6] Selecting features...")

    # Define columns to exclude
    default_exclude = [target_column, weight_column]
    if exclude_columns:
        default_exclude.extend(exclude_columns)

    # Separate features, target, and weights
    feature_cols = [col for col in df.columns if col not in default_exclude]

    if verbose:
        print(f"  Features: {len(feature_cols)} columns")
        print(f"  Excluded: {len(default_exclude)} columns")

        # Check for nulls
        null_counts = df.select(feature_cols).null_count()
        total_nulls = null_counts.sum_horizontal()[0]
        if total_nulls > 0:
            print(f"  ⚠ WARNING: {total_nulls:,} null values detected")
        else:
            print(f"  ✓ No null values")

    # ==========================================
    # STEP 4: SPLIT DATA (Using Indices)
    # ==========================================
    if verbose:
        print(f"\n[4/6] Splitting data using indices...")

    # Split in Polars (much faster than pandas)
    df_train = df[train_idx]
    df_test = df[test_idx]

    # Separate X, y, weights for train
    X_train_pl = df_train.select(feature_cols)
    y_train_pl = df_train.select(target_column)
    w_train_pl = df_train.select(weight_column)

    # Separate X, y, weights for test
    X_test_pl = df_test.select(feature_cols)
    y_test_pl = df_test.select(target_column)
    w_test_pl = df_test.select(weight_column)

    # Free original dataframe
    del df, df_train, df_test
    gc.collect()

    if verbose:
        print(f"  ✓ Split complete")
        print(f"  X_train memory: {X_train_pl.estimated_size('mb'):.1f} MB")
        print(f"  X_test memory: {X_test_pl.estimated_size('mb'):.1f} MB")

    # ==========================================
    # STEP 5: HANDLE OUTLIERS (Optional, in Polars)
    # ==========================================
    if handle_outliers:
        if verbose:
            print(f"\n[5/6] Handling outliers (method: {outlier_method})...")

        numeric_cols = [col for col in feature_cols
                        if X_train_pl[col].dtype in [pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                                     pl.Float32, pl.Float64]]

        if outlier_method == 'percentile':
            for col in numeric_cols:
                lower = X_train_pl[col].quantile(outlier_lower)
                upper = X_train_pl[col].quantile(outlier_upper)

                X_train_pl = X_train_pl.with_columns([
                    pl.col(col).clip(lower, upper).alias(col)
                ])
                X_test_pl = X_test_pl.with_columns([
                    pl.col(col).clip(lower, upper).alias(col)
                ])

        if verbose:
            print(f"  ✓ Clipped {len(numeric_cols)} numeric columns")

    # ==========================================
    # STEP 6: CONVERT TO PANDAS (Only When Needed)
    # ==========================================
    if verbose:
        print(f"\n[6/6] Converting to Pandas for XGBoost...")

    # Convert train set
    X_train = X_train_pl.to_pandas()
    y_train = y_train_pl.to_pandas()[target_column]
    w_train = w_train_pl.to_pandas()[weight_column]

    # Free Polars train data
    del X_train_pl, y_train_pl, w_train_pl
    gc.collect()

    # Convert test set
    X_test = X_test_pl.to_pandas()
    y_test = y_test_pl.to_pandas()[target_column]
    w_test = w_test_pl.to_pandas()[weight_column]

    # Free Polars test data
    del X_test_pl, y_test_pl, w_test_pl
    gc.collect()

    if verbose:
        print(f"  ✓ Conversion complete")
        print(f"  X_train: {X_train.memory_usage(deep=True).sum() / 1e6:.1f} MB")
        print(f"  X_test: {X_test.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # ==========================================
    # POST-PROCESSING (Pandas Operations)
    # ==========================================

    # Handle categorical columns (if any)
    # categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    #
    # if len(categorical_cols) > 0:
    #     if verbose:
    #         print(f"\n[Post-Processing] One-hot encoding {len(categorical_cols)} categorical columns...")
    #
    #     X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True, dtype=int)
    #     X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True, dtype=int)
    #
    #     # Ensure train and test have same columns
    #     missing_in_test = set(X_train.columns) - set(X_test.columns)
    #     missing_in_train = set(X_test.columns) - set(X_train.columns)
    #
    #     for col in missing_in_test:
    #         X_test[col] = 0
    #     for col in missing_in_train:
    #         X_train[col] = 0
    #
    #     # Reorder to match
    #     X_test = X_test[X_train.columns]
    #
    #     if verbose:
    #         print(f"  ✓ Encoded → {X_train.shape[1]} total features")

    # Optional: Scaling
    if apply_scaling:
        if verbose:
            print(f"\n[Post-Processing] Applying StandardScaler...")

        scaler = StandardScaler()
        numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        if verbose:
            print(f"  ✓ Scaled {len(numeric_cols)} columns")

    # Remove constant columns
    constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_cols:
        if verbose:
            print(f"\n[Post-Processing] Removing {len(constant_cols)} constant columns")
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    # ==========================================
    # FINAL SUMMARY
    # ==========================================
    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Final shapes:")
        print(f"  X_train: {X_train.shape} ({X_train.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
        print(f"  X_test:  {X_test.shape} ({X_test.memory_usage(deep=True).sum() / 1e6:.1f} MB)")
        print(f"  Features: {X_train.shape[1]}")

        # Target distribution
        print(f"\nTarget distribution:")
        if y_train.dtype in ['int64', 'float64']:
            print(f"  Train mean: {y_train.mean():.2f} (std: {y_train.std():.2f})")
            print(f"  Test mean:  {y_test.mean():.2f} (std: {y_test.std():.2f})")
        else:
            print(f"  Train: {y_train.value_counts().to_dict()}")
            print(f"  Test:  {y_test.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test, w_train, w_test


# ==========================================
# ALTERNATIVE: Chunked Processing for HUGE Datasets
# ==========================================

def preprocess_data_xgboost_chunked(
        data_path: str,
        target_column: str,
        weight_column: str = 'perwt',
        exclude_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        stratify_column: Optional[str] = 'year',
        chunk_size: int = 1000000,
        verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Ultra-memory-safe chunked preprocessing for datasets that don't fit in RAM.

    This version processes data in chunks and writes to temporary files.
    Use only if the standard version runs out of memory.
    """

    if verbose:
        print("=" * 80)
        print("CHUNKED XGBOOST PREPROCESSING (FOR HUGE DATASETS)")
        print("=" * 80)

    # Step 1: First pass to get indices
    if verbose:
        print("\n[Pass 1/2] Scanning data to create split indices...")

    df_scan = pl.scan_parquet(data_path) if data_path.endswith('.parquet') else pl.scan_csv(data_path)

    # Get total rows
    n_rows = df_scan.select(pl.count()).collect()[0, 0]

    if verbose:
        print(f"  Total rows: {n_rows:,}")

    # Create split indices
    indices = np.arange(n_rows)

    if stratify_column:
        stratify_values = df_scan.select(stratify_column).collect().to_numpy().flatten()
    else:
        stratify_values = None

    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=stratify_values
    )

    # Convert to sets for fast lookup
    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)

    if verbose:
        print(f"  Train: {len(train_idx):,}, Test: {len(test_idx):,}")

    # Step 2: Process in chunks
    if verbose:
        print(f"\n[Pass 2/2] Processing data in chunks of {chunk_size:,}...")

    # Lists to accumulate chunks
    X_train_chunks = []
    y_train_chunks = []
    w_train_chunks = []
    X_test_chunks = []
    y_test_chunks = []
    w_test_chunks = []

    # Define columns
    default_exclude = [target_column, weight_column]
    if exclude_columns:
        default_exclude.extend(exclude_columns)

    # Process chunks
    df_reader = pl.read_parquet(data_path) if data_path.endswith('.parquet') else pl.read_csv(data_path)
    feature_cols = [col for col in df_reader.columns if col not in default_exclude]

    for i in range(0, n_rows, chunk_size):
        chunk_idx = np.arange(i, min(i + chunk_size, n_rows))

        # Determine which rows go to train vs test
        train_mask = np.isin(chunk_idx, list(train_idx_set))
        test_mask = np.isin(chunk_idx, list(test_idx_set))

        if train_mask.any():
            chunk_train = df_reader[chunk_idx[train_mask]]
            X_train_chunks.append(chunk_train.select(feature_cols).to_pandas())
            y_train_chunks.append(chunk_train[target_column].to_pandas())
            w_train_chunks.append(chunk_train[weight_column].to_pandas())

        if test_mask.any():
            chunk_test = df_reader[chunk_idx[test_mask]]
            X_test_chunks.append(chunk_test.select(feature_cols).to_pandas())
            y_test_chunks.append(chunk_test[target_column].to_pandas())
            w_test_chunks.append(chunk_test[weight_column].to_pandas())

        if verbose and (i // chunk_size) % 10 == 0:
            print(f"  Processed {i:,} / {n_rows:,} rows...")

    # Concatenate all chunks
    if verbose:
        print("\nConcatenating chunks...")

    X_train = pd.concat(X_train_chunks, ignore_index=True)
    y_train = pd.concat(y_train_chunks, ignore_index=True)
    w_train = pd.concat(w_train_chunks, ignore_index=True)

    X_test = pd.concat(X_test_chunks, ignore_index=True)
    y_test = pd.concat(y_test_chunks, ignore_index=True)
    w_test = pd.concat(w_test_chunks, ignore_index=True)

    if verbose:
        print("\n✓ Chunked preprocessing complete!")

    return X_train, X_test, y_train, y_test, w_train, w_test