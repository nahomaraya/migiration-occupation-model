import xgboost as xgb
import pandas as pd
import numpy as np
from typing import Dict

from models.clustering import perform_clustering

def analyze_high_error_clusters(
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        t_eval: pd.Series,
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        error_percentile: float = 0.95,
        max_clusters: int = 10,
        use_full_clustering: bool = True,
        verbose: bool = True
) -> Dict:
    """
    Step 4: Cluster workers with high prediction errors.

    Uses your existing perform_clustering() function for full analysis
    including K-Means, Hierarchical, silhouette scores, and visualizations.

    These are people whose occupational success CANNOT be explained
    by our variables - potentially interesting subgroups.
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 4: CLUSTER HIGH-ERROR CASES")
        print("=" * 80)
        print("\n  Goal: Find subgroups where model failed badly")

    # Create analysis dataframe
    df = X_eval.copy()
    df['y'] = y_eval.values
    df['t'] = t_eval.values
    df['residual_y'] = residuals_y
    df['residual_t'] = residuals_t
    df['abs_error'] = np.abs(residuals_y)

    # Filter to high-error cases (top 10% errors)
    threshold = df['abs_error'].quantile(error_percentile)
    high_error = df[df['abs_error'] > threshold].copy()

    if verbose:
        print(f"\n  Error threshold (top {(1 - error_percentile) * 100:.0f}%): {threshold:.2f}")
        print(f"  High-error cases: {len(high_error):,}")

    # Select numeric features for clustering (exclude metadata columns)
    exclude_from_clustering = ['y', 't', 'residual_y', 'residual_t', 'abs_error']
    numeric_cols = high_error.select_dtypes(include=[np.number]).columns.tolist()
    cluster_cols = [c for c in numeric_cols if c not in exclude_from_clustering]

    if len(cluster_cols) < 2 or len(high_error) < 100:
        if verbose:
            print("  ⚠ Insufficient data for clustering")
        return {'high_error_cases': high_error, 'cluster_results': None}

    # Limit features for clustering (top 20 by variance)
    feature_variance = high_error[cluster_cols].var().sort_values(ascending=False)
    top_cluster_cols = feature_variance.head(20).index.tolist()

    if verbose:
        print(f"  Using top {len(top_cluster_cols)} features for clustering")

    # Prepare data for clustering
    X_cluster = high_error[top_cluster_cols].fillna(0)

    # Split for clustering (80/20 for train/val as your clustering code expects)
    from sklearn.model_selection import train_test_split
    X_cluster_train, X_cluster_val = train_test_split(
        X_cluster, test_size=0.2, random_state=42
    )

    if use_full_clustering and len(X_cluster_train) >= 100:
        if verbose:
            print("\n  Running full clustering analysis with your perform_clustering()...")
            print("  " + "-" * 60)

        # Use your existing clustering function!
        cluster_results = perform_clustering(
            x_train=X_cluster_train,
            x_val=X_cluster_val,
            y_train=None,  # Unsupervised
            y_val=None,
            max_clusters=max_clusters
        )

        # Add cluster labels back to high_error dataframe
        # Re-cluster full dataset with optimal K
        from sklearn.cluster import KMeans
        optimal_k = cluster_results['optimal_k']
        kmeans_full = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        high_error['cluster'] = kmeans_full.fit_predict(X_cluster)

    else:
        # Fallback to simple MiniBatchKMeans if dataset too small
        if verbose:
            print("\n  Using simplified clustering (dataset too small for full analysis)")

        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=4, random_state=42, n_init=3)
        high_error['cluster'] = kmeans.fit_predict(X_cluster)

        cluster_results = {
            'optimal_k': 4,
            'kmeans_model': kmeans,
            'metrics': {'kmeans': {'silhouette': None}}
        }

    # Analyze cluster profiles with immigration context
    if verbose:
        print("\n" + "=" * 80)
        print("CLUSTER PROFILES (Immigration Analysis)")
        print("=" * 80)

    optimal_k = cluster_results['optimal_k']

    for cluster_id in range(optimal_k):
        cluster_data = high_error[high_error['cluster'] == cluster_id]
        n = len(cluster_data)

        if n == 0:
            continue

        immigrant_pct = cluster_data['t'].mean() * 100
        avg_score = cluster_data['y'].mean()
        avg_residual = cluster_data['residual_y'].mean()

        if verbose:
            print(f"\n  Cluster {cluster_id} (n={n:,}, {n / len(high_error) * 100:.1f}% of high-error cases):")
            print(f"    Immigrant %: {immigrant_pct:.1f}%")
            print(f"    Avg OCCSCORE: {avg_score:.1f}")
            print(f"    Avg Residual: {avg_residual:+.2f}")

            # Interpretation
            if avg_residual > 5:
                if immigrant_pct > 50:
                    print("    → IMMIGRANT OVERACHIEVERS: Doing much better than predicted!")
                else:
                    print("    → NATIVE OVERACHIEVERS: Doing much better than predicted!")
            elif avg_residual < -5:
                if immigrant_pct > 50:
                    print("    → IMMIGRANT UNDERACHIEVERS: Doing worse than predicted")
                else:
                    print("    → NATIVE UNDERACHIEVERS: Doing worse than predicted")
            else:
                print("    → MODERATE ERROR: Model errors are average")

            # Top distinguishing features
            print(f"    Top features:")
            for col in top_cluster_cols[:3]:
                val = cluster_data[col].mean()
                print(f"      - {col}: {val:.2f}")

    return {
        'high_error_cases': high_error,
        'cluster_results': cluster_results,
        'threshold': threshold,
        'optimal_k': cluster_results['optimal_k'],
        'cluster_features': top_cluster_cols
    }

