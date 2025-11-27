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
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from typing import Dict, Optional, Tuple
import warnings

from dml_pipeline.plot_casual_results import plot_ate_results, export_ate_latex, plot_regression_diagnostics

warnings.filterwarnings('ignore')


def run_leakage_debug(
        data_path: str,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        sample_size: int = 10000,
        exclude_cols: list = None,
        mlflow_experiment: str = "dml_leakage_debug"
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

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run():
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("sample_size", sample_size)
        mlflow.log_param("treatment_col", treatment_col)
        mlflow.log_param("outcome_col", outcome_col)
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
        print("STEP 10: Running Causal Estimation Debug...")
        causal_results = debug_causal_estimation(
            residuals_y=residuals_y,
            residuals_t=residuals_t,
            weights=w_eval.values,
            t_eval=t_eval,
            y_eval=y_eval
        )
        cluster_results = debug_high_error_clustering(
            X_eval=X_eval,
            y_eval=y_eval,
            t_eval=t_eval,
            residuals_y=residuals_y,  # from Step 1
            residuals_t=residuals_t,  # from Step 2
            w_eval=w_eval,
            error_percentile=0.90,  # top 10% errors
            max_clusters=8,
            verbose=True
        )

        plot_ate_results(
            causal_results,
            treatment_name="Immigration",
            outcome_name="OCCSCORE",
            save_path="ate_forest_plot.png"
        )

        diagnostics = plot_regression_diagnostics(
            causal_results=causal_results,
            residuals_y=residuals_y,
            residuals_t=residuals_t,
            save_path="regression_diagnostics.png",
            show_plot=True
        )

        # Access diagnostic test results
        print(f"Breusch-Pagan p-value: {diagnostics['breusch_pagan']['p_value']:.4f}")
        print(f"Jarque-Bera p-value: {diagnostics['jarque_bera']['p_value']:.4f}")
        print(f"Durbin-Watson: {diagnostics['durbin_watson']['statistic']:.4f}")

        # Export to LaTeX for thesis
        export_ate_latex(causal_results, "thesis_tables/ate_table.tex")

        return causal_results, cluster_results




def debug_high_error_clustering(
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        t_eval: pd.Series,
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        w_eval: Optional[pd.Series] = None,
        error_percentile: float = 0.90,
        max_clusters: int = 8,
        verbose: bool = True
) -> Dict:
    """
    Debug clustering analysis on high-error residual cases.

    Args:
        X_eval: Features for evaluation set
        y_eval: Outcome values (OCCSCORE)
        t_eval: Treatment values (is_immigrant)
        residuals_y: Outcome residuals from Step 1
        residuals_t: Treatment residuals from Step 2
        w_eval: Sample weights (optional)
        error_percentile: Percentile threshold (0.90 = top 10% errors)
        max_clusters: Maximum clusters to try
        verbose: Print diagnostics

    Returns:
        Dictionary with clustering results and diagnostics
    """

    print("=" * 80)
    print("STEP 4: DEBUG CLUSTERING ON HIGH-ERROR RESIDUALS")
    print("=" * 80)

    # ========================================
    # 4.1: Create Analysis DataFrame
    # ========================================
    print("\n" + "-" * 60)
    print("4.1: Preparing High-Error Cases")
    print("-" * 60)

    # Build analysis dataframe
    df = X_eval.copy()
    df['y_actual'] = y_eval.values
    df['t_actual'] = t_eval.values
    df['residual_y'] = residuals_y
    df['residual_t'] = residuals_t
    df['abs_error'] = np.abs(residuals_y)

    if w_eval is not None:
        df['weight'] = w_eval.values

    # Calculate error threshold
    threshold = df['abs_error'].quantile(error_percentile)

    print(f"\n  Total samples: {len(df):,}")
    print(f"  Error threshold ({(1 - error_percentile) * 100:.0f}% worst): {threshold:.2f}")

    # Filter to high-error cases
    high_error = df[df['abs_error'] > threshold].copy()
    low_error = df[df['abs_error'] <= threshold].copy()

    print(f"  High-error cases: {len(high_error):,} ({len(high_error) / len(df) * 100:.1f}%)")
    print(f"  Low-error cases: {len(low_error):,} ({len(low_error) / len(df) * 100:.1f}%)")

    # ========================================
    # 4.2: Compare High vs Low Error Groups
    # ========================================
    print("\n" + "-" * 60)
    print("4.2: High-Error vs Low-Error Comparison")
    print("-" * 60)

    print(f"\n  {'Metric':<30} {'High Error':>15} {'Low Error':>15} {'Diff':>10}")
    print("  " + "-" * 70)

    # Compare key metrics
    metrics_to_compare = [
        ('Avg OCCSCORE', 'y_actual'),
        ('Avg Residual (Y)', 'residual_y'),
        ('Immigrant %', 't_actual'),
        ('Avg |Error|', 'abs_error'),
    ]

    for name, col in metrics_to_compare:
        high_val = high_error[col].mean()
        low_val = low_error[col].mean()
        diff = high_val - low_val

        if col == 't_actual':
            print(f"  {name:<30} {high_val * 100:>14.1f}% {low_val * 100:>14.1f}% {diff * 100:>+9.1f}%")
        else:
            print(f"  {name:<30} {high_val:>15.2f} {low_val:>15.2f} {diff:>+10.2f}")

    # Check if high-error cases are mostly over or under predictions
    over_predicted = (high_error['residual_y'] < 0).sum()
    under_predicted = (high_error['residual_y'] > 0).sum()

    print(f"\n  High-Error Breakdown:")
    print(
        f"    Over-predicted (actual < predicted):  {over_predicted:,} ({over_predicted / len(high_error) * 100:.1f}%)")
    print(
        f"    Under-predicted (actual > predicted): {under_predicted:,} ({under_predicted / len(high_error) * 100:.1f}%)")

    # ========================================
    # 4.3: Select Features for Clustering
    # ========================================
    print("\n" + "-" * 60)
    print("4.3: Selecting Features for Clustering")
    print("-" * 60)

    # Exclude non-feature columns
    exclude_cols = ['y_actual', 't_actual', 'residual_y', 'residual_t', 'abs_error', 'weight']
    feature_cols = [c for c in high_error.columns if c not in exclude_cols]

    # Keep only numeric columns
    numeric_cols = high_error[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

    print(f"\n  Available features: {len(feature_cols)}")
    print(f"  Numeric features: {len(numeric_cols)}")

    # Check for missing values
    missing_pct = high_error[numeric_cols].isnull().mean()
    high_missing = missing_pct[missing_pct > 0.1]

    if len(high_missing) > 0:
        print(f"\n  ⚠ Features with >10% missing:")
        for col, pct in high_missing.items():
            print(f"      {col}: {pct * 100:.1f}%")

    # Select top features by variance
    feature_variance = high_error[numeric_cols].var().sort_values(ascending=False)
    top_features = feature_variance.head(20).index.tolist()

    print(f"\n  Selected top {len(top_features)} features by variance:")
    for i, feat in enumerate(top_features[:10], 1):
        print(f"    {i:2d}. {feat:<30} (var: {feature_variance[feat]:.2f})")
    if len(top_features) > 10:
        print(f"    ... and {len(top_features) - 10} more")

    # Prepare clustering data
    X_cluster = high_error[top_features].fillna(0).copy()

    # Standardize for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    X_scaled_df = pd.DataFrame(X_scaled, columns=top_features, index=high_error.index)

    print(f"\n  ✓ Prepared {len(X_cluster):,} samples × {len(top_features)} features")

    # ========================================
    # 4.4: Determine Optimal K
    # ========================================
    print("\n" + "-" * 60)
    print("4.4: Determining Optimal Number of Clusters")
    print("-" * 60)

    K_range = range(2, max_clusters + 1)
    inertias = []
    silhouette_scores = []
    davies_bouldin_scores = []

    print(f"\n  {'K':>3} {'Inertia':>12} {'Silhouette':>12} {'Davies-Bouldin':>15}")
    print("  " + "-" * 45)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        inertias.append(kmeans.inertia_)
        sil_score = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)

        silhouette_scores.append(sil_score)
        davies_bouldin_scores.append(db_score)

        print(f"  {k:>3} {kmeans.inertia_:>12.1f} {sil_score:>12.4f} {db_score:>15.4f}")

    # Find optimal K
    optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
    optimal_k_db = K_range[np.argmin(davies_bouldin_scores)]

    print(f"\n  Optimal K (Silhouette): {optimal_k_silhouette}")
    print(f"  Optimal K (Davies-Bouldin): {optimal_k_db}")

    # Use silhouette-based optimal K
    optimal_k = optimal_k_silhouette
    print(f"\n  → Using K = {optimal_k}")

    # ========================================
    # 4.5: Plot Elbow and Silhouette
    # ========================================
    print("\n" + "-" * 60)
    print("4.5: Plotting Cluster Selection Metrics")
    print("-" * 60)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Elbow plot
    axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[0].set_xlabel('Number of Clusters (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Silhouette plot
    axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[1].set_xlabel('Number of Clusters (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Score (Higher = Better)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Davies-Bouldin plot
    axes[2].plot(K_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
    axes[2].axvline(x=optimal_k, color='b', linestyle='--', label=f'Optimal K={optimal_k}')
    axes[2].set_xlabel('Number of Clusters (K)')
    axes[2].set_ylabel('Davies-Bouldin Score')
    axes[2].set_title('Davies-Bouldin Score (Lower = Better)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cluster_selection_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("  ✓ Saved: cluster_selection_metrics.png")

    # ========================================
    # 4.6: Fit Final Clustering Model
    # ========================================
    print("\n" + "-" * 60)
    print("4.6: Fitting Final Clustering Models")
    print("-" * 60)

    # K-Means
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    high_error['cluster_kmeans'] = kmeans_final.fit_predict(X_scaled)

    # Hierarchical
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    high_error['cluster_hier'] = hierarchical.fit_predict(X_scaled)

    # Metrics
    kmeans_silhouette = silhouette_score(X_scaled, high_error['cluster_kmeans'])
    hier_silhouette = silhouette_score(X_scaled, high_error['cluster_hier'])

    print(f"\n  K-Means Silhouette: {kmeans_silhouette:.4f}")
    print(f"  Hierarchical Silhouette: {hier_silhouette:.4f}")

    # Use better performing method
    if kmeans_silhouette >= hier_silhouette:
        high_error['cluster'] = high_error['cluster_kmeans']
        best_method = 'K-Means'
    else:
        high_error['cluster'] = high_error['cluster_hier']
        best_method = 'Hierarchical'

    print(f"\n  → Using {best_method} (higher silhouette)")

    # ========================================
    # 4.7: Cluster Profiles (THE KEY OUTPUT)
    # ========================================
    print("\n" + "-" * 60)
    print("4.7: Cluster Profiles")
    print("-" * 60)

    print("\n" + "=" * 80)
    print("CLUSTER PROFILES - HIGH ERROR CASES")
    print("=" * 80)

    cluster_profiles = []

    for cluster_id in range(optimal_k):
        cluster_data = high_error[high_error['cluster'] == cluster_id]
        n = len(cluster_data)
        pct = n / len(high_error) * 100

        # Key metrics
        avg_occscore = cluster_data['y_actual'].mean()
        avg_residual = cluster_data['residual_y'].mean()
        immigrant_pct = cluster_data['t_actual'].mean() * 100
        avg_abs_error = cluster_data['abs_error'].mean()

        # Direction of error
        over_pred_pct = (cluster_data['residual_y'] < 0).mean() * 100
        under_pred_pct = (cluster_data['residual_y'] > 0).mean() * 100

        # Store profile
        profile = {
            'cluster': cluster_id,
            'n': n,
            'pct': pct,
            'avg_occscore': avg_occscore,
            'avg_residual': avg_residual,
            'immigrant_pct': immigrant_pct,
            'avg_abs_error': avg_abs_error,
            'over_predicted_pct': over_pred_pct,
            'under_predicted_pct': under_pred_pct
        }
        cluster_profiles.append(profile)

        # Determine cluster type
        if avg_residual > 5:
            cluster_type = "OVERACHIEVERS"
            emoji = "📈"
        elif avg_residual < -5:
            cluster_type = "UNDERACHIEVERS"
            emoji = "📉"
        else:
            cluster_type = "MODERATE ERROR"
            emoji = "📊"

        # Immigration context
        if immigrant_pct > 60:
            immig_type = "Mostly Immigrants"
        elif immigrant_pct < 40:
            immig_type = "Mostly Natives"
        else:
            immig_type = "Mixed"

        print(f"""
  │ CLUSTER {cluster_id}: {cluster_type} {emoji}
  │ {immig_type} ({immigrant_pct:.1f}% immigrant)

  │ Size:           {n:,} ({pct:.1f}% of high-error cases)
  │ Avg OCCSCORE:   {avg_occscore:.1f}
  │ Avg Residual:   {avg_residual:+.2f}
  │ Avg |Error|:    {avg_abs_error:.2f}
  │ Over-predicted: {over_pred_pct:.1f}%
  │ Under-predicted:{under_pred_pct:.1f}%
 """)

        # Top distinguishing features for this cluster
        print(f"\n  Top Features (vs overall mean):")
        cluster_means = cluster_data[top_features].mean()
        overall_means = high_error[top_features].mean()
        diff = (cluster_means - overall_means).abs().sort_values(ascending=False)

        for feat in diff.head(5).index:
            cluster_val = cluster_means[feat]
            overall_val = overall_means[feat]
            direction = "↑" if cluster_val > overall_val else "↓"
            print(f"      {direction} {feat}: {cluster_val:.2f} (overall: {overall_val:.2f})")

    # ========================================
    # 4.8: Heterogeneous Treatment Effects
    # ========================================
    print("\n" + "-" * 60)
    print("4.8: Heterogeneous Treatment Effects by Cluster")
    print("-" * 60)

    print("\n  Testing if causal effect differs across clusters...")
    print(f"\n  {'Cluster':>8} {'N':>8} {'Immig %':>10} {'Avg Resid Y':>12} {'Avg Resid T':>12} {'Implied Effect':>15}")
    print("  " + "-" * 70)

    import statsmodels.api as sm

    for cluster_id in range(optimal_k):
        cluster_data = high_error[high_error['cluster'] == cluster_id]
        n = len(cluster_data)

        res_y = cluster_data['residual_y'].values
        res_t = cluster_data['residual_t'].values
        immig_pct = cluster_data['t_actual'].mean() * 100

        # Simple correlation as proxy for effect
        if np.std(res_t) > 0.01:  # Avoid division by zero
            # Run mini OLS within cluster
            X_ols = sm.add_constant(res_t)
            model = sm.OLS(res_y, X_ols).fit()
            effect = model.params[1]
            pval = model.pvalues[1]

            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(
                f"  {cluster_id:>8} {n:>8,} {immig_pct:>9.1f}% {np.mean(res_y):>+12.2f} {np.mean(res_t):>+12.4f} {effect:>+14.2f}{sig}")
        else:
            print(
                f"  {cluster_id:>8} {n:>8,} {immig_pct:>9.1f}% {np.mean(res_y):>+12.2f} {np.mean(res_t):>+12.4f} {'N/A':>15}")

    print("\n  * p<0.05, ** p<0.01, *** p<0.001")

    # ========================================
    # 4.9: PCA Visualization
    # ========================================
    print("\n" + "-" * 60)
    print("4.9: PCA Visualization of Clusters")
    print("-" * 60)

    # 2D PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n  PCA Explained Variance: {pca.explained_variance_ratio_.sum() * 100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Color by cluster
    scatter1 = axes[0].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=high_error['cluster'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[0].set_title('Clusters (High-Error Cases)')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')

    # Color by residual (over vs under prediction)
    scatter2 = axes[1].scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=high_error['residual_y'],
        cmap='RdYlGn',
        alpha=0.6,
        s=30,
        vmin=-20, vmax=20
    )
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[1].set_title('Residuals (Green=Over-achieving, Red=Under-achieving)')
    plt.colorbar(scatter2, ax=axes[1], label='Residual Y')

    plt.tight_layout()
    plt.savefig('cluster_pca_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("  ✓ Saved: cluster_pca_visualization.png")

    # ========================================
    # 4.10: Summary
    # ========================================
    print("\n" + "=" * 80)
    print("CLUSTERING DEBUG SUMMARY")
    print("=" * 80)

    # Create summary dataframe
    profiles_df = pd.DataFrame(cluster_profiles)

    print(f"""
  │ HIGH-ERROR CLUSTERING RESULTS                               │
  │ Total high-error cases:  {len(high_error):,}
  │ Optimal clusters:        {optimal_k}
  │ Method:                  {best_method}
  │ Silhouette score:        {kmeans_silhouette:.4f}
    """)

    # Identify interesting clusters
    overachievers = profiles_df[profiles_df['avg_residual'] > 5]
    underachievers = profiles_df[profiles_df['avg_residual'] < -5]

    if len(overachievers) > 0:
        print("  📈 OVERACHIEVER CLUSTERS (doing better than predicted):")
        for _, row in overachievers.iterrows():
            print(
                f"      Cluster {int(row['cluster'])}: {row['immigrant_pct']:.0f}% immigrant, residual +{row['avg_residual']:.1f}")

    if len(underachievers) > 0:
        print("\n  📉 UNDERACHIEVER CLUSTERS (doing worse than predicted):")
        for _, row in underachievers.iterrows():
            print(
                f"      Cluster {int(row['cluster'])}: {row['immigrant_pct']:.0f}% immigrant, residual {row['avg_residual']:.1f}")

    print("\n  For thesis: Examine these clusters to understand WHO the model")
    print("  fails to predict — may reveal heterogeneous treatment effects.")

    return {
        'high_error_df': high_error,
        'low_error_df': low_error,
        'cluster_profiles': profiles_df,
        'optimal_k': optimal_k,
        'kmeans_model': kmeans_final,
        'silhouette_score': kmeans_silhouette,
        'pca_model': pca,
        'scaler': scaler,
        'feature_cols': top_features,
        'threshold': threshold
    }

def debug_causal_estimation(
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        weights: np.ndarray = None,
        t_eval: pd.Series = None,
        y_eval: pd.Series = None
):
    """
    Debug Step 3: Analyze the causal estimation regression.

    Checks:
    1. Residual distributions look reasonable
    2. No extreme outliers distorting results
    3. Compares naive vs DML estimates
    4. Validates regression assumptions
    """
    import statsmodels.api as sm
    from scipy import stats

    print("\n" + "=" * 80)
    print("STEP 10: DEBUG CAUSAL ESTIMATION")
    print("=" * 80)

    # ----------------------------------------
    # 10a: Check residual distributions
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10a: Residual Distributions")
    print("-" * 60)

    print(f"\n  Outcome Residuals (Y - Ŷ):")
    print(f"    Mean:   {np.mean(residuals_y):.4f} (should be ~0)")
    print(f"    Std:    {np.std(residuals_y):.4f}")
    print(f"    Min:    {np.min(residuals_y):.4f}")
    print(f"    Max:    {np.max(residuals_y):.4f}")
    print(f"    Median: {np.median(residuals_y):.4f}")

    # Check if mean is close to zero
    if abs(np.mean(residuals_y)) > 1:
        print(f"    ⚠ WARNING: Mean residual is not close to 0!")
    else:
        print(f"    ✓ Mean residual looks good")

    print(f"\n  Treatment Residuals (T - T̂):")
    print(f"    Mean:   {np.mean(residuals_t):.4f} (should be ~0)")
    print(f"    Std:    {np.std(residuals_t):.4f}")
    print(f"    Min:    {np.min(residuals_t):.4f}")
    print(f"    Max:    {np.max(residuals_t):.4f}")
    print(f"    Median: {np.median(residuals_t):.4f}")

    if abs(np.mean(residuals_t)) > 0.1:
        print(f"    ⚠ WARNING: Mean residual is not close to 0!")
    else:
        print(f"    ✓ Mean residual looks good")

    # ----------------------------------------
    # 10b: Check for extreme outliers
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10b: Outlier Analysis")
    print("-" * 60)

    # Outcome residuals outliers
    y_q1, y_q99 = np.percentile(residuals_y, [1, 99])
    y_outliers = np.sum((residuals_y < y_q1) | (residuals_y > y_q99))
    y_outlier_pct = y_outliers / len(residuals_y) * 100

    print(f"\n  Outcome residuals:")
    print(f"    1st percentile:  {y_q1:.4f}")
    print(f"    99th percentile: {y_q99:.4f}")
    print(f"    Outliers (outside 1-99%): {y_outliers:,} ({y_outlier_pct:.1f}%)")

    # Treatment residuals outliers
    t_q1, t_q99 = np.percentile(residuals_t, [1, 99])
    t_outliers = np.sum((residuals_t < t_q1) | (residuals_t > t_q99))
    t_outlier_pct = t_outliers / len(residuals_t) * 100

    print(f"\n  Treatment residuals:")
    print(f"    1st percentile:  {t_q1:.4f}")
    print(f"    99th percentile: {t_q99:.4f}")
    print(f"    Outliers (outside 1-99%): {t_outliers:,} ({t_outlier_pct:.1f}%)")

    # ----------------------------------------
    # 10c: Compare Naive vs DML Estimate
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10c: Naive vs DML Comparison")
    print("-" * 60)

    if t_eval is not None and y_eval is not None:
        # Naive estimate: simple difference in means
        y_immigrant = y_eval[t_eval == 1].mean()
        y_native = y_eval[t_eval == 0].mean()
        naive_effect = y_immigrant - y_native

        print(f"\n  NAIVE ESTIMATE (simple mean difference):")
        print(f"    Avg OCCSCORE (immigrants): {y_immigrant:.2f}")
        print(f"    Avg OCCSCORE (natives):    {y_native:.2f}")
        print(f"    Naive Effect: {naive_effect:.4f}")

        # Naive OLS (no controls)
        X_naive = sm.add_constant(t_eval.values)
        naive_model = sm.OLS(y_eval.values, X_naive)
        naive_results = naive_model.fit()
        naive_ols_effect = naive_results.params[1]
        naive_pvalue = naive_results.pvalues[1]

        print(f"\n  NAIVE OLS (Y ~ T, no controls):")
        print(f"    Coefficient: {naive_ols_effect:.4f}")
        print(f"    P-value: {naive_pvalue:.6f}")

    # DML estimate
    X_dml = sm.add_constant(residuals_t)
    if weights is not None:
        dml_model = sm.WLS(residuals_y, X_dml, weights=weights)
    else:
        dml_model = sm.OLS(residuals_y, X_dml)

    dml_results = dml_model.fit()
    dml_effect = dml_results.params[1]
    dml_se = dml_results.bse[1]
    dml_pvalue = dml_results.pvalues[1]
    dml_ci = dml_results.conf_int()[1]

    print(f"\n  DML ESTIMATE (residualized regression):")
    print(f"    Coefficient: {dml_effect:.4f}")
    print(f"    Std Error:   {dml_se:.4f}")
    print(f"    P-value:     {dml_pvalue:.6f}")
    print(f"    95% CI:      [{dml_ci[0]:.4f}, {dml_ci[1]:.4f}]")

    # Compare
    if t_eval is not None and y_eval is not None:
        bias_removed = naive_ols_effect - dml_effect
        print(f"\n  COMPARISON:")
        print(f"    Naive effect:     {naive_ols_effect:.4f}")
        print(f"    DML effect:       {dml_effect:.4f}")
        print(f"    Bias removed:     {bias_removed:.4f}")
        print(f"    Bias direction:   {'Naive overestimates' if bias_removed > 0 else 'Naive underestimates'}")

        if abs(bias_removed) > 1:
            print(f"\n    ✓ DML removed substantial confounding bias!")
        else:
            print(f"\n    ℹ Confounding bias was relatively small")

    # ----------------------------------------
    # 10d: Regression Diagnostics
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10d: Regression Diagnostics")
    print("-" * 60)

    # Heteroskedasticity test (Breusch-Pagan)
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(dml_results.resid, X_dml)

    print(f"\n  Breusch-Pagan Test (Heteroskedasticity):")
    print(f"    Statistic: {bp_stat:.4f}")
    print(f"    P-value:   {bp_pvalue:.6f}")
    if bp_pvalue < 0.05:
        print(f"    ⚠ Heteroskedasticity detected - using robust SEs recommended")
    else:
        print(f"    ✓ No significant heteroskedasticity")

    # Normality of residuals (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(dml_results.resid)

    print(f"\n  Jarque-Bera Test (Normality):")
    print(f"    Statistic: {jb_stat:.4f}")
    print(f"    P-value:   {jb_pvalue:.6f}")
    if jb_pvalue < 0.05:
        print(f"    ⚠ Residuals not normally distributed (common with large samples)")
    else:
        print(f"    ✓ Residuals approximately normal")

    # R-squared (should be LOW for DML - we've removed confounders)
    print(f"\n  Model Fit:")
    print(f"    R-squared: {dml_results.rsquared:.4f}")
    if dml_results.rsquared > 0.3:
        print(f"    ⚠ R² is surprisingly high for residualized regression")
    else:
        print(f"    ✓ Low R² is expected (confounders already removed)")

    # ----------------------------------------
    # 10e: Robust Standard Errors
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10e: Robust Standard Errors Comparison")
    print("-" * 60)

    # HC3 robust standard errors
    dml_robust = dml_model.fit(cov_type='HC3')
    robust_se = dml_robust.bse[1]
    robust_pvalue = dml_robust.pvalues[1]
    robust_ci = dml_robust.conf_int()[1]

    print(f"\n  Standard OLS:")
    print(f"    SE: {dml_se:.4f}, P-value: {dml_pvalue:.6f}")
    print(f"    95% CI: [{dml_ci[0]:.4f}, {dml_ci[1]:.4f}]")

    print(f"\n  Robust (HC3):")
    print(f"    SE: {robust_se:.4f}, P-value: {robust_pvalue:.6f}")
    print(f"    95% CI: [{robust_ci[0]:.4f}, {robust_ci[1]:.4f}]")

    se_diff = (robust_se - dml_se) / dml_se * 100
    print(f"\n  SE difference: {se_diff:+.1f}%")
    if abs(se_diff) > 20:
        print(f"    ⚠ Large difference - use robust SEs for inference")
    else:
        print(f"    ✓ Similar SEs - either approach is fine")

    # ----------------------------------------
    # 10f: Full OLS Summary
    # ----------------------------------------
    print("\n" + "-" * 60)
    print("10f: Full OLS Summary (for thesis)")
    print("-" * 60)
    print(dml_results.summary())

    # ----------------------------------------
    # Final Summary
    # ----------------------------------------
    print("\n" + "=" * 80)
    print("CAUSAL ESTIMATION DEBUG SUMMARY")
    print("=" * 80)

    print(f"\n  Causal Effect of Immigration on OCCSCORE:")

    print(f"  ║  Effect:     {dml_effect:>8.4f}                  ║")
    print(f"  ║  Std Error:  {dml_se:>8.4f}                  ║")
    print(f"  ║  t-stat:     {dml_results.tvalues[1]:>8.4f}                  ║")
    print(f"  ║  P-value:    {dml_pvalue:>8.6f}                ║")
    print(f"  ║  95% CI:     [{dml_ci[0]:.4f}, {dml_ci[1]:.4f}]       ║")

    # Interpretation
    stars = "***" if dml_pvalue < 0.001 else "**" if dml_pvalue < 0.01 else "*" if dml_pvalue < 0.05 else ""

    print(f"\n  For thesis table: {dml_effect:.4f}{stars} ({dml_se:.4f})")

    if dml_pvalue < 0.05:
        if dml_effect < 0:
            print(f"\n  ✓ SIGNIFICANT NEGATIVE EFFECT")
            print(f"    Immigrants have {abs(dml_effect):.2f} lower occupational scores")
            print(f"    after controlling for confounders.")
        else:
            print(f"\n  ✓ SIGNIFICANT POSITIVE EFFECT")
            print(f"    Immigrants have {dml_effect:.2f} higher occupational scores")
            print(f"    after controlling for confounders.")
    else:
        print(f"\n  ✗ NO SIGNIFICANT EFFECT (p={dml_pvalue:.4f})")
        print(f"    Cannot conclude immigration affects occupational scores")
        print(f"    after controlling for confounders.")

    return {
        'dml_effect': dml_effect,
        'dml_se': dml_se,
        'dml_pvalue': dml_pvalue,
        'dml_ci': (dml_ci[0], dml_ci[1]),
        'robust_se': robust_se,
        'robust_pvalue': robust_pvalue,
        'naive_effect': naive_ols_effect if t_eval is not None else None,
        'bias_removed': bias_removed if t_eval is not None else None,
        'ols_results': dml_results,
        'ols_robust': dml_robust
    }
# =========================================
# QUICK RUN
# =========================================

