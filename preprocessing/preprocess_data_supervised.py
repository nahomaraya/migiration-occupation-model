from typing import Tuple, Optional, List
import warnings
import pandas as pd
import polars as pl
import numpy as np
from preprocessing.clip_outliers  import clip_outliers as  clip_outliers_iqr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess_data_supervised(
        data_path: str,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        apply_scaling: bool = True,
        handle_outliers: bool = True,
        verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    General supervised learning preprocessing (for non-XGBoost models).

    Returns train/val/test splits with scaling applied.

    Args:
        data_path: Path to data file
        target_column: Name of target variable
        test_size: Proportion of test set
        val_size: Proportion of validation set (from remaining data)
        apply_scaling: Whether to apply StandardScaler
        handle_outliers: Whether to clip outliers
        verbose: Print diagnostic information

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

    if verbose:
        print("=" * 80)
        print("SUPERVISED LEARNING PREPROCESSING")
        print("=" * 80)

    # Load data
    if data_path.endswith('.parquet'):
        data = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        data = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        data = pd.read_excel(data_path)
    else:
        raise ValueError("File must be .parquet, .csv, or .xlsx")

    if verbose:
        print(f"Loaded: {data.shape[0]:,} rows × {data.shape[1]} columns")

    # Clean column names
    data.columns = data.columns.str.strip()
    data.columns = data.columns.str.replace(r"\s+", " ", regex=True)

    # Handle missing values
    if verbose:
        missing_counts = data.isnull().sum()
        if missing_counts.sum() > 0:
            print(f"\nMissing values found: {missing_counts.sum()}")
        data.fillna(data.mean(numeric_only=True), inplace=True)

    # Handle outliers
    if handle_outliers:
        if verbose:
            print("\nClipping outliers to 5th-95th percentile...")
        for col in data.select_dtypes(include=['int64', 'float64']).columns:
            if col != target_column:
                data[col] = data[col].clip(
                    lower=data[col].quantile(0.05),
                    upper=data[col].quantile(0.95)
                )

    # Separate features and target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # One-hot encode categorical features
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        if verbose:
            print(f"\nOne-hot encoding {len(categorical_cols)} categorical columns")
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

    # Apply scaling
    if apply_scaling:
        if verbose:
            print("\nApplying StandardScaler...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42
    )

    val_proportion = val_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - val_proportion), random_state=42
    )

    # Remove constant and highly correlated columns
    constant_cols = [col for col in X_train.columns if X_train[col].nunique() <= 1]
    if constant_cols:
        X_train = X_train.drop(columns=constant_cols)
        X_val = X_val.drop(columns=[c for c in constant_cols if c in X_val.columns])
        X_test = X_test.drop(columns=[c for c in constant_cols if c in X_test.columns])

    corr_matrix = X_train.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    if to_drop:
        X_train = X_train.drop(columns=to_drop)
        X_val = X_val.drop(columns=[c for c in to_drop if c in X_val.columns])
        X_test = X_test.drop(columns=[c for c in to_drop if c in X_test.columns])

    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"Train: {X_train.shape[0]:,} samples, {X_train.shape[1]} features")
        print(f"Val:   {X_val.shape[0]:,} samples")
        print(f"Test:  {X_test.shape[0]:,} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test