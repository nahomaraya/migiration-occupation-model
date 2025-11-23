from typing import Tuple, Optional, List
import warnings
import pandas as pd
import polars as pl
import numpy as np
from preprocessing.clip_outliers  import clip_outliers as  clip_outliers_iqr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data_xgboost(
        data_path: str,
        target_column: str,
        weight_column: str = 'perwt',
        exclude_columns: Optional[List[str]] = None,
        test_size: float = 0.2,
        stratify_column: Optional[str] = 'year',
        apply_scaling: bool = False,
        handle_outliers: bool = False,
        outlier_method: str = 'percentile',  # 'percentile' or 'iqr'
        outlier_lower: float = 0.01,
        outlier_upper: float = 0.99,
        verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Preprocess data for XGBoost training.

    XGBoost-specific optimizations:
    - No mandatory scaling (tree-based models don't need it)
    - Handles categorical variables natively (if using enable_categorical=True)
    - Preserves weights for sample weighting
    - Optional outlier handling

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
        outlier_lower: Lower percentile for clipping (if percentile method)
        outlier_upper: Upper percentile for clipping (if percentile method)
        verbose: Print diagnostic information

    Returns:
        X_train, X_test, y_train, y_test, w_train, w_test
    """

    if verbose:
        print("=" * 80)
        print("XGBOOST DATA PREPROCESSING")
        print("=" * 80)

    # Load data
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
        if verbose:
            print(f"✓ Loaded parquet file: {data_path}")
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
        if verbose:
            print(f"✓ Loaded CSV file: {data_path}")
    else:
        raise ValueError("File must be .parquet or .csv")

    if verbose:
        print(f"Shape: {data.shape[0]:,} rows × {data.shape[1]} columns")
        print(f"Memory: {data.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Validate required columns
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    if weight_column not in data.columns:
        raise ValueError(f"Weight column '{weight_column}' not found in data")

    # Define columns to exclude from features
    default_exclude = [target_column, weight_column]
    if exclude_columns:
        default_exclude.extend(exclude_columns)

    # Check for stratification column
    stratify_data = None
    if stratify_column and stratify_column in data.columns:
        stratify_data = data[stratify_column]
        if verbose:
            print(f"✓ Will stratify split by: {stratify_column}")

    # Separate features, target, and weights
    feature_cols = [col for col in data.columns if col not in default_exclude]

    X = data[feature_cols].copy()
    y = data[target_column].copy()
    weights = data[weight_column].copy()

    if verbose:
        print(f"\nFeatures: {len(feature_cols)} columns")
        print(f"Target: {target_column}")
        print(f"Weights: {weight_column}")

        # Check for missing values
        null_counts = X.isnull().sum()
        if null_counts.sum() > 0:
            print("\n⚠ WARNING: Missing values detected!")
            print(null_counts[null_counts > 0])
        else:
            print("\n✓ No missing values")

        # Target distribution
        print(f"\nTarget distribution:")
        if y.dtype in ['int64', 'float64']:
            print(f"  Mean: {y.mean():.2f}")
            print(f"  Std: {y.std():.2f}")
            print(f"  Min: {y.min():.2f}")
            print(f"  Max: {y.max():.2f}")
        else:
            print(y.value_counts())

    # Handle outliers if requested
    if handle_outliers:
        if verbose:
            print(f"\n[Outlier Handling] Method: {outlier_method}")

        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            if outlier_method == 'percentile':
                lower = X[col].quantile(outlier_lower)
                upper = X[col].quantile(outlier_upper)
                X[col] = X[col].clip(lower=lower, upper=upper)
            elif outlier_method == 'iqr':
                clip_outliers_iqr(X, col)

        if verbose:
            print(f"  ✓ Clipped {len(numeric_cols)} numeric columns")

    # Handle categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if len(categorical_cols) > 0:
        if verbose:
            print(f"\n[Categorical Encoding] Found {len(categorical_cols)} categorical columns")
            print(f"  Columns: {categorical_cols}")

        # One-hot encode
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

        if verbose:
            print(f"  ✓ One-hot encoded → {X.shape[1]} total features")

    # Optional: Scaling (usually not needed for XGBoost)
    if apply_scaling:
        if verbose:
            print("\n[Scaling] Applying StandardScaler...")

        scaler = StandardScaler()
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        if verbose:
            print(f"  ✓ Scaled {len(numeric_cols)} numeric columns")

    # Train-test split
    if verbose:
        print(f"\n[Train-Test Split] Test size: {test_size * 100:.0f}%")

    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights,
        test_size=test_size,
        random_state=42,
        stratify=stratify_data
    )

    if verbose:
        print(f"  Train: {X_train.shape[0]:,} samples ({X_train.shape[0] / len(X) * 100:.1f}%)")
        print(f"  Test:  {X_test.shape[0]:,} samples ({X_test.shape[0] / len(X) * 100:.1f}%)")

        # Check target distribution in splits
        print(f"\n  Target distribution:")
        if y.dtype in ['int64', 'float64']:
            print(f"    Train mean: {y_train.mean():.2f}")
            print(f"    Test mean:  {y_test.mean():.2f}")
        else:
            print(f"    Train: {y_train.value_counts(normalize=True).to_dict()}")
            print(f"    Test:  {y_test.value_counts(normalize=True).to_dict()}")

    # Remove constant columns (based on training set only)
    constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_cols:
        if verbose:
            print(f"\n[Feature Cleaning] Removing {len(constant_cols)} constant columns")
        X_train = X_train.drop(columns=constant_cols)
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    # Optional: Remove highly correlated features (usually not needed for XGBoost)
    # Uncomment if you want this:
    # corr_matrix = X_train.corr(numeric_only=True).abs()
    # upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    # if to_drop and verbose:
    #     print(f"[Feature Cleaning] Removing {len(to_drop)} highly correlated columns (>0.95)")
    #     X_train = X_train.drop(columns=to_drop)
    #     X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Final shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test:  {X_test.shape}")
        print(f"  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, w_train, w_test

