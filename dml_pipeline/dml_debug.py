# dml_debug.py
"""
DML Debug Script - Check for Data Leakage
==========================================

Run this on a tiny sample (10k rows) before running the full pipeline.
Checks:
1. Treatment column (is_immigrant) is NOT in features
2. Outcome column (occscore) is NOT in features
3. No highly correlated proxies for treatment/outcome
4. Feature distributions look reasonable
5. Models train without errors
"""

import polars as pl
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def run_leakage_debug(
        data_path: str,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        sample_size: int = 10000,
        exclude_cols: list = None
):
    """
    Debug DML pipeline on tiny sample to check for data leakage.

    Args:
        data_path: Path to parquet file
        treatment_col: Treatment variable
        outcome_col: Outcome variable
        weight_col: Weight variable
        sample_size: Number of rows to sample (default 10k)
        exclude_cols: Additional columns to exclude
    """

    print("=" * 80)
    print("DML DEBUG SCRIPT - DATA LEAKAGE CHECK")
    print("=" * 80)
    print(f"\nSample size: {sample_size:,} rows")
    print(f"Treatment: {treatment_col}")
    print(f"Outcome: {outcome_col}")

    # =========================================
    # STEP 1: Load tiny sample
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 1: Loading tiny sample...")
    print("-" * 60)

    df = pl.read_parquet(data_path)
    df_sample = df.sample(n=min(sample_size, len(df)), seed=42)
    df_pd = df_sample.to_pandas()

    print(f"  ✓ Loaded {len(df_pd):,} rows, {len(df_pd.columns)} columns")

    del df, df_sample

    # =========================================
    # STEP 2: Check required columns exist
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 2: Checking required columns...")
    print("-" * 60)

    missing = []
    for col in [treatment_col, outcome_col, weight_col]:
        if col in df_pd.columns:
            print(f"  ✓ Found: {col}")
        else:
            print(f"  ✗ MISSING: {col}")
            missing.append(col)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # =========================================
    # STEP 3: Define feature columns
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 3: Defining feature columns...")
    print("-" * 60)

    # Columns that MUST be excluded
    must_exclude = [treatment_col, outcome_col, weight_col]

    # Standard ID/metadata columns to exclude
    standard_exclude = [
        'year', 'serial', 'pernum', 'hhwt', 'cluster', 'strata',
        'gq', 'sample', 'cbserial', 'histid', 'multyear', 'sploc',
        'sprule', 'perwt', 'citizenship_status', 'origin_development_level', 'origin_region',
        'bpld', 'citizenship_status','age_at_arrival','years_in_us',
        'immigrant_x_education', 'bpl','immigrant_x_english',
        'yrnatur','yrimmig','is_naturalized','immigrant_x_stem', 'tenure_x_education'
    ]

    # User-specified exclusions
    if exclude_cols:
        standard_exclude.extend(exclude_cols)

    # Build final exclusion list
    all_exclude = list(set(must_exclude + standard_exclude))

    # Feature columns
    feature_cols = [c for c in df_pd.columns if c not in all_exclude]

    print(f"  Total columns: {len(df_pd.columns)}")
    print(f"  Excluded: {len(all_exclude)}")
    print(f"  Features: {len(feature_cols)}")

    # =========================================
    # STEP 4: CRITICAL - Check for leakage
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 4: CHECKING FOR DATA LEAKAGE")
    print("-" * 60)

    leakage_found = False

    # Check 4a: Treatment not in features
    if treatment_col in feature_cols:
        print(f"  ✗ LEAKAGE: {treatment_col} is in features!")
        leakage_found = True
    else:
        print(f"  ✓ {treatment_col} correctly excluded from features")

    # Check 4b: Outcome not in features
    if outcome_col in feature_cols:
        print(f"  ✗ LEAKAGE: {outcome_col} is in features!")
        leakage_found = True
    else:
        print(f"  ✓ {outcome_col} correctly excluded from features")

    # Check 4c: Check for highly correlated columns with treatment
    print(f"\n  Checking correlations with {treatment_col}...")

    X = df_pd[feature_cols].select_dtypes(include=[np.number])
    treatment_corr = X.corrwith(df_pd[treatment_col]).abs().sort_values(ascending=False)

    high_corr_treatment = treatment_corr[treatment_corr > 0.8]
    if len(high_corr_treatment) > 0:
        print(f"  ⚠ WARNING: High correlation with {treatment_col}:")
        for col, corr in high_corr_treatment.items():
            print(f"      {col}: {corr:.3f}")
            if corr > 0.95:
                print(f"      ✗ LIKELY LEAKAGE: {col} may be a proxy for treatment!")
                leakage_found = True
    else:
        print(f"  ✓ No features highly correlated (>0.8) with {treatment_col}")

    # Check 4d: Check for highly correlated columns with outcome
    print(f"\n  Checking correlations with {outcome_col}...")

    outcome_corr = X.corrwith(df_pd[outcome_col]).abs().sort_values(ascending=False)

    high_corr_outcome = outcome_corr[outcome_corr > 0.8]
    if len(high_corr_outcome) > 0:
        print(f"  ⚠ WARNING: High correlation with {outcome_col}:")
        for col, corr in high_corr_outcome.items():
            print(f"      {col}: {corr:.3f}")
            if corr > 0.95:
                print(f"      ✗ LIKELY LEAKAGE: {col} may be a proxy for outcome!")
                leakage_found = True
    else:
        print(f"  ✓ No features highly correlated (>0.8) with {outcome_col}")

    # Check 4e: Look for suspicious column names
    print(f"\n  Checking for suspicious column names...")

    suspicious_patterns = ['immigrant', 'native', 'foreign', 'born', 'occscore', 'occ_score', 'sei', 'hwsei']
    suspicious_found = []

    for col in feature_cols:
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                suspicious_found.append((col, pattern))

    if suspicious_found:
        print(f"  ⚠ WARNING: Suspicious column names found:")
        for col, pattern in suspicious_found:
            print(f"      {col} (contains '{pattern}')")
        print(f"  → Review these columns to ensure no leakage!")
    else:
        print(f"  ✓ No suspicious column names detected")

    # =========================================
    # STEP 5: Summary of features
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 5: Feature Summary")
    print("-" * 60)

    print(f"\n  Feature columns ({len(feature_cols)}):")
    for i, col in enumerate(feature_cols[:20]):
        dtype = df_pd[col].dtype
        nunique = df_pd[col].nunique()
        null_pct = df_pd[col].isnull().mean() * 100
        print(f"    {i + 1:2d}. {col:<30} dtype={str(dtype):<10} unique={nunique:<8} null={null_pct:.1f}%")

    if len(feature_cols) > 20:
        print(f"    ... and {len(feature_cols) - 20} more features")

    # =========================================
    # STEP 6: Test train-test split
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 6: Testing train-test split (50/50 for DML)...")
    print("-" * 60)

    X = df_pd[feature_cols]
    y = df_pd[outcome_col]
    t = df_pd[treatment_col]
    w = df_pd[weight_col]

    X_train, X_eval, y_train, y_eval, t_train, t_eval, w_train, w_eval = train_test_split(
        X, y, t, w,
        test_size=0.5,
        random_state=42,
        stratify=t
    )

    print(f"  ✓ Train: {len(X_train):,} rows")
    print(f"  ✓ Eval:  {len(X_eval):,} rows")
    print(f"  ✓ Treatment balance (train): {t_train.mean() * 100:.1f}% immigrant")
    print(f"  ✓ Treatment balance (eval):  {t_eval.mean() * 100:.1f}% immigrant")

    # =========================================
    # STEP 7: Test Outcome Model (Y ~ X)
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 7: Testing Outcome Model (Y ~ X)...")
    print("-" * 60)

    try:
        dtrain_y = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
        deval_y = xgb.DMatrix(X_eval, label=y_eval, enable_categorical=True)

        params_y = {
            'objective': 'reg:squarederror',
            'max_depth': 4,
            'eta': 0.3,
            'seed': 42
        }

        model_y = xgb.train(params_y, dtrain_y, num_boost_round=10, verbose_eval=False)
        y_pred = model_y.predict(deval_y)

        residuals_y = y_eval.values - y_pred
        rmse = np.sqrt(np.mean(residuals_y ** 2))
        r2 = 1 - (np.sum(residuals_y ** 2) / np.sum((y_eval - y_eval.mean()) ** 2))

        print(f"  ✓ Model trained successfully!")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R²: {r2:.4f}")

        # Check for suspiciously high R²
        if r2 > 0.95:
            print(f"  ⚠ WARNING: R² is suspiciously high ({r2:.4f})")
            print(f"    This may indicate data leakage!")
            leakage_found = True
        elif r2 > 0.8:
            print(f"  ⚠ NOTE: R² is quite high - double-check features")
        else:
            print(f"  ✓ R² looks reasonable for this task")

    except Exception as e:
        print(f"  ✗ ERROR training outcome model: {e}")
        return False

    # =========================================
    # STEP 8: Test Treatment Model (T ~ X)
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 8: Testing Treatment Model (T ~ X)...")
    print("-" * 60)

    try:
        dtrain_t = xgb.DMatrix(X_train, label=t_train, weight=w_train, enable_categorical=True)
        deval_t = xgb.DMatrix(X_eval, label=t_eval, enable_categorical=True)

        params_t = {
            'objective': 'binary:logistic',
            'max_depth': 4,
            'eta': 0.3,
            'seed': 42
        }

        model_t = xgb.train(params_t, dtrain_t, num_boost_round=10, verbose_eval=False)
        t_pred_proba = model_t.predict(deval_t)
        t_pred = (t_pred_proba > 0.5).astype(int)

        residuals_t = t_eval.values - t_pred_proba
        accuracy = np.mean(t_pred == t_eval.values)

        print(f"  ✓ Model trained successfully!")
        print(f"    Accuracy: {accuracy:.4f}")

        # Check for suspiciously high accuracy
        if accuracy > 0.95:
            print(f"  ⚠ WARNING: Accuracy is suspiciously high ({accuracy:.4f})")
            print(f"    This may indicate data leakage!")
            leakage_found = True
        elif accuracy > 0.85:
            print(f"  ⚠ NOTE: Accuracy is quite high - double-check features")
        else:
            print(f"  ✓ Accuracy looks reasonable")

    except Exception as e:
        print(f"  ✗ ERROR training treatment model: {e}")
        return False

    # =========================================
    # STEP 9: Check feature importance for leaks
    # =========================================
    print("\n" + "-" * 60)
    print("STEP 9: Checking feature importance for leaks...")
    print("-" * 60)

    # Outcome model feature importance
    importance_y = model_y.get_score(importance_type='gain')
    if importance_y:
        top_features_y = sorted(importance_y.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 features for OUTCOME model:")
        for feat, imp in top_features_y:
            print(f"    {feat}: {imp:.2f}")
            # Check if any look suspicious
            if any(p in feat.lower() for p in suspicious_patterns):
                print(f"      ⚠ This feature name looks suspicious!")

    # Treatment model feature importance
    importance_t = model_t.get_score(importance_type='gain')
    if importance_t:
        top_features_t = sorted(importance_t.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  Top 5 features for TREATMENT model:")
        for feat, imp in top_features_t:
            print(f"    {feat}: {imp:.2f}")
            if any(p in feat.lower() for p in suspicious_patterns):
                print(f"      ⚠ This feature name looks suspicious!")

    # =========================================
    # FINAL VERDICT
    # =========================================
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if leakage_found:
        print("\n  ✗ POTENTIAL DATA LEAKAGE DETECTED!")
        print("    Review the warnings above before running full pipeline.")
        print("    Common fixes:")
        print("    - Remove columns that are proxies for treatment/outcome")
        print("    - Check for post-treatment variables")
        print("    - Verify feature engineering doesn't use future information")
        return False
    else:
        print("\n  ✓ NO OBVIOUS DATA LEAKAGE DETECTED")
        print("    Pipeline appears safe to run on full dataset.")
        print("\n    Recommended next step:")
        print("    >>> results = run_dml_from_file('your_data.parquet', sample_fraction=0.1)")
        return True


# =========================================
# QUICK RUN
# =========================================

