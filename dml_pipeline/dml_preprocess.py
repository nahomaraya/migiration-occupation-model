import pandas as pd
import polars as pl
import numpy as np
import gc
from typing import Dict, List, Optional, Tuple

from preprocessing.preprocess_data import preprocess_data_xgboost
from sklearn.model_selection import train_test_split

def preprocess_for_dml(
        data_path: str,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        exclude_cols: Optional[List[str]] = None,
        verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series,
pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Preprocess data for DML using existing preprocess_data_xgboost.

    Key differences from standard preprocessing:
    1. Uses 50/50 split (cross-fitting requirement)
    2. Excludes treatment from features but preserves it
    3. Returns both outcome (Y) and treatment (T) for both splits

    Args:
        data_path: Path to sampled parquet file
        treatment_col: Treatment variable (is_immigrant)
        outcome_col: Outcome variable (occscore)
        weight_col: Weight variable (perwt)
        exclude_cols: Additional columns to exclude
        verbose: Print progress

    Returns:
        X_train, X_eval, y_train, y_eval, t_train, t_eval, w_train, w_eval
    """
    if verbose:
        print("\n" + "=" * 80)
        print("PREPROCESSING FOR DML")
        print("=" * 80)

    # Build exclusion list (treatment must be excluded from features!)
    dml_exclude = [treatment_col]  # Don't use treatment as a feature
    if exclude_cols:
        dml_exclude.extend(exclude_cols)

    # Also exclude identifiers that shouldn't be features
    standard_exclude = ['year', 'serial', 'pernum', 'hhwt', 'cluster',
                        'strata', 'gq', 'sample', 'cbserial', 'histid',
                        'multyear', 'sploc', 'sprule']
    dml_exclude.extend([c for c in standard_exclude if c not in dml_exclude])

    if verbose:
        print(f"\n  Treatment: {treatment_col}")
        print(f"  Outcome: {outcome_col}")
        print(f"  Excluding from features: {len(dml_exclude)} columns")

    # Use existing preprocessor with 50/50 split for cross-fitting
    X_train, X_eval, y_train, y_eval, w_train, w_eval = preprocess_data_xgboost(
        data_path=data_path,
        target_column=outcome_col,
        weight_column=weight_col,
        exclude_columns=dml_exclude,
        test_size=0.5,  # DML requires 50/50 split!
        stratify_column=treatment_col,  # Balance treatment across folds
        apply_scaling=False,  # XGBoost doesn't need scaling
        handle_outliers=False,  # Keep raw values for interpretability
        verbose=verbose
    )

    # Now we need to get treatment values for both splits
    # Load just the treatment column using the same indices
    if verbose:
        print(f"\n  Loading treatment variable ({treatment_col})...")

    df_treatment = pl.read_parquet(data_path, columns=[treatment_col])

    # We need to match indices - the preprocessor used the same random state
    # So we recreate the split indices
    n_samples = len(df_treatment)
    indices = np.arange(n_samples)

    # Get stratification values
    stratify_values = df_treatment[treatment_col].to_numpy()

    # Recreate the exact same split
    train_idx, eval_idx = train_test_split(
        indices,
        test_size=0.5,
        random_state=42,
        stratify=stratify_values
    )

    # Extract treatment values
    t_full = df_treatment[treatment_col].to_pandas()
    t_train = t_full.iloc[train_idx].reset_index(drop=True)
    t_eval = t_full.iloc[eval_idx].reset_index(drop=True)

    # Also reset index on other series to match
    y_train = y_train.reset_index(drop=True)
    y_eval = y_eval.reset_index(drop=True)
    w_train = w_train.reset_index(drop=True)
    w_eval = w_eval.reset_index(drop=True)

    del df_treatment, t_full
    gc.collect()

    if verbose:
        print(f"\n  Split Summary (50/50 for DML cross-fitting):")
        print(f"    Train: {len(X_train):,} samples")
        print(f"    Eval:  {len(X_eval):,} samples")
        print(f"    Treatment rate (train): {t_train.mean() * 100:.1f}%")
        print(f"    Treatment rate (eval):  {t_eval.mean() * 100:.1f}%")

    return X_train, X_eval, y_train, y_eval, t_train, t_eval, w_train, w_eval
