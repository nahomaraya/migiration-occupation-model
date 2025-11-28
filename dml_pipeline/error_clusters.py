import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from typing import Dict, Optional


def high_error_clustering(
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        t_eval: pd.Series,
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        w_eval: Optional[pd.Series] = None,
        error_percentile: float = 0.90,
        max_clusters: int = 8,
        sample_size_clustering: int = 50000,
        sample_size_plot: int = 5000,
        verbose: bool = True,
        random_seed: int = 42
) -> Dict:
    """
    Memory-optimized clustering analysis on high-error residual cases.
    Uses sampling for large datasets.
    """

    rng = np.random.default_rng(random_seed)

    print("=" * 70)
    print("STEP 4: HIGH-ERROR CLUSTERING (OPTIMIZED)")
    print("=" * 70)

    # ========================================
    # 4.1: Identify High-Error Cases (memory efficient)
    # ========================================
    print("\n4.1: Identifying High-Error Cases")
    print("-" * 50)

    n_total = len(y_eval)
    abs_errors = np.abs(residuals_y)
    threshold = np.percentile(abs_errors, error_percentile * 100)
    high_error_mask = abs_errors > threshold
    n_high_error = high_error_mask.sum()

    print(f"  Total samples: {n_total:,}")
    print(f"  Error threshold (top {(1 - error_percentile) * 100:.0f}%): {threshold:.2f}")
    print(f"  High-error cases: {n_high_error:,}")

    # Get indices of high-error cases
    high_error_idx = np.where(high_error_mask)[0]

    # ========================================
    # 4.2: Sample if needed (KEY OPTIMIZATION)
    # ========================================
    if n_high_error > sample_size_clustering:
        print(f"\n  Sampling {sample_size_clustering:,} from {n_high_error:,} high-error cases")
        sample_idx = rng.choice(high_error_idx, sample_size_clustering, replace=False)
        using_sample = True
    else:
        sample_idx = high_error_idx
        using_sample = False

    # Build minimal dataframe (only what we need)
    high_error = pd.DataFrame({
        'y_actual': y_eval.iloc[sample_idx].values,
        't_actual': t_eval.iloc[sample_idx].values,
        'residual_y': residuals_y[sample_idx],
        'residual_t': residuals_t[sample_idx],
        'abs_error': abs_errors[sample_idx]
    }, index=sample_idx)

    # ========================================
    # 4.3: Compare High vs Low Error (on full data, no df needed)
    # ========================================
    print("\n4.2: High vs Low Error Comparison")
    print("-" * 50)

    low_error_mask = ~high_error_mask

    comparisons = [
        ('Avg Outcome', y_eval.values[high_error_mask].mean(), y_eval.values[low_error_mask].mean()),
        ('Avg |Error|', abs_errors[high_error_mask].mean(), abs_errors[low_error_mask].mean()),
        ('Treatment %', t_eval.values[high_error_mask].mean() * 100, t_eval.values[low_error_mask].mean() * 100),
    ]

    print(f"\n  {'Metric':<20} {'High Error':>12} {'Low Error':>12} {'Diff':>10}")
    print("  " + "-" * 55)
    for name, high_val, low_val in comparisons:
        print(f"  {name:<20} {high_val:>12.2f} {low_val:>12.2f} {high_val - low_val:>+10.2f}")

    over_pred = (residuals_y[high_error_mask] < 0).sum()
    under_pred = (residuals_y[high_error_mask] > 0).sum()
    print(f"\n  Over-predicted: {over_pred:,} ({over_pred / n_high_error * 100:.1f}%)")
    print(f"  Under-predicted: {under_pred:,} ({under_pred / n_high_error * 100:.1f}%)")

    # ========================================
    # 4.4: Select Features for Clustering
    # ========================================
    print("\n4.3: Feature Selection")
    print("-" * 50)

    # Get numeric features only
    X_sample = X_eval.iloc[sample_idx]
    numeric_cols = X_sample.select_dtypes(include=[np.number]).columns.tolist()

    # Fast variance calculation on sample
    variances = X_sample[numeric_cols].var()
    top_features = variances.nlargest(20).index.tolist()

    print(f"  Using top {len(top_features)} features by variance")

    # Prepare clustering data
    X_cluster = X_sample[top_features].fillna(0).values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    print(f"  Prepared: {X_scaled.shape[0]:,} samples × {X_scaled.shape[1]} features")

    # ========================================
    # 4.5: Find Optimal K (use MiniBatchKMeans for speed)
    # ========================================
    print("\n4.4: Finding Optimal K")
    print("-" * 50)

    K_range = range(2, max_clusters + 1)
    silhouette_scores = []

    # Use MiniBatchKMeans for large datasets
    use_minibatch = len(X_scaled) > 10000

    print(f"\n  {'K':>3} {'Silhouette':>12} {'Inertia':>12}")
    print("  " + "-" * 30)

    for k in K_range:
        if use_minibatch:
            model = MiniBatchKMeans(n_clusters=k, random_state=random_seed,
                                    batch_size=1024, n_init=3)
        else:
            model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)

        labels = model.fit_predict(X_scaled)

        # Silhouette on subsample for speed
        if len(X_scaled) > 10000:
            sil_idx = rng.choice(len(X_scaled), 10000, replace=False)
            sil = silhouette_score(X_scaled[sil_idx], labels[sil_idx])
        else:
            sil = silhouette_score(X_scaled, labels)

        silhouette_scores.append(sil)
        print(f"  {k:>3} {sil:>12.4f} {model.inertia_:>12.1f}")

    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n  → Optimal K = {optimal_k}")

    # ========================================
    # 4.6: Fit Final Model
    # ========================================
    print("\n4.5: Fitting Final Model")
    print("-" * 50)

    if use_minibatch:
        kmeans_final = MiniBatchKMeans(n_clusters=optimal_k, random_state=random_seed,
                                       batch_size=1024, n_init=5)
    else:
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=random_seed, n_init=10)

    high_error['cluster'] = kmeans_final.fit_predict(X_scaled)

    final_silhouette = silhouette_scores[optimal_k - 2]
    print(f"  Silhouette Score: {final_silhouette:.4f}")

    # ========================================
    # 4.7: Cluster Profiles
    # ========================================
    print("\n4.6: Cluster Profiles")
    print("=" * 70)

    cluster_profiles = []

    for cid in range(optimal_k):
        mask = high_error['cluster'] == cid
        cd = high_error[mask]
        n = len(cd)

        profile = {
            'cluster': cid,
            'n': n,
            'pct': n / len(high_error) * 100,
            'avg_outcome': cd['y_actual'].mean(),
            'avg_residual': cd['residual_y'].mean(),
            'treatment_pct': cd['t_actual'].mean() * 100,
            'avg_abs_error': cd['abs_error'].mean(),
            'over_predicted_pct': (cd['residual_y'] < 0).mean() * 100
        }
        cluster_profiles.append(profile)

        # Determine type
        if profile['avg_residual'] > 5:
            ctype, emoji = "OVERACHIEVERS", "📈"
        elif profile['avg_residual'] < -5:
            ctype, emoji = "UNDERACHIEVERS", "📉"
        else:
            ctype, emoji = "MODERATE", "📊"

        print(f"""
  CLUSTER {cid}: {ctype} {emoji}
  ├─ Size: {n:,} ({profile['pct']:.1f}%)
  ├─ Avg Outcome: {profile['avg_outcome']:.1f}
  ├─ Avg Residual: {profile['avg_residual']:+.2f}
  ├─ Treatment %: {profile['treatment_pct']:.1f}%
  └─ Over-predicted: {profile['over_predicted_pct']:.1f}%""")

        # Top features for this cluster
        cluster_feat_means = X_sample.loc[cd.index, top_features].mean()
        overall_feat_means = X_sample[top_features].mean()
        diffs = (cluster_feat_means - overall_feat_means).abs().nlargest(3)

        print("      Top features vs mean:")
        for feat in diffs.index:
            cv, ov = cluster_feat_means[feat], overall_feat_means[feat]
            arrow = "↑" if cv > ov else "↓"
            print(f"        {arrow} {feat}: {cv:.2f} (avg: {ov:.2f})")

    profiles_df = pd.DataFrame(cluster_profiles)

    # ========================================
    # 4.8: Heterogeneous Effects by Cluster
    # ========================================
    print("\n" + "-" * 50)
    print("4.7: Treatment Effects by Cluster")
    print("-" * 50)

    import statsmodels.api as sm

    print(f"\n  {'Cluster':>7} {'N':>7} {'Effect':>10} {'SE':>8} {'p-val':>10}")
    print("  " + "-" * 45)

    for cid in range(optimal_k):
        cd = high_error[high_error['cluster'] == cid]
        res_y = cd['residual_y'].values
        res_t = cd['residual_t'].values

        if np.std(res_t) > 0.01:
            X_ols = sm.add_constant(res_t)
            fit = sm.OLS(res_y, X_ols).fit(cov_type='HC1')
            eff, se, pval = fit.params[1], fit.bse[1], fit.pvalues[1]
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {cid:>7} {len(cd):>7,} {eff:>+10.3f} {se:>8.3f} {pval:>10.4f} {sig}")
        else:
            print(f"  {cid:>7} {len(cd):>7,} {'N/A':>10}")

    # ========================================
    # 4.9: Visualizations (SAMPLED)
    # ========================================
    print("\n" + "-" * 50)
    print("4.8: Generating Plots")
    print("-" * 50)

    # Sample for plotting
    if len(high_error) > sample_size_plot:
        plot_idx = rng.choice(len(high_error), sample_size_plot, replace=False)
    else:
        plot_idx = np.arange(len(high_error))

    X_plot = X_scaled[plot_idx]
    clusters_plot = high_error['cluster'].values[plot_idx]
    residuals_plot = high_error['residual_y'].values[plot_idx]

    # PCA for visualization
    pca = PCA(n_components=2, random_state=random_seed)
    X_pca = pca.fit_transform(X_plot)

    exp_var = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA explained variance: {exp_var:.1f}%")

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Plot 1: Silhouette scores
    axes[0].plot(list(K_range), silhouette_scores, 'o-', color='#4878A8', lw=2, ms=8)
    axes[0].axvline(optimal_k, color='#D55E00', ls='--', lw=1.5, label=f'K={optimal_k}')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].set_title('(A) Cluster Selection')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: PCA by cluster
    scatter1 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=clusters_plot,
                               cmap='viridis', alpha=0.5, s=15, rasterized=True)
    axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[1].set_title('(B) Clusters')
    plt.colorbar(scatter1, ax=axes[1], label='Cluster')

    # Plot 3: PCA by residual
    scatter2 = axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=residuals_plot,
                               cmap='RdYlGn', alpha=0.5, s=15,
                               vmin=-20, vmax=20, rasterized=True)
    axes[2].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)')
    axes[2].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)')
    axes[2].set_title('(C) Residuals')
    plt.colorbar(scatter2, ax=axes[2], label='Residual')

    # Add note about sampling
    if using_sample or len(high_error) > sample_size_plot:
        note = f'Note: Plots show {len(plot_idx):,} sampled points from {n_high_error:,} high-error cases'
        fig.text(0.5, 0.01, note, ha='center', fontsize=8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig('cluster_analysis.png', dpi=150, bbox_inches='tight')

    if verbose:
        plt.show()
    plt.close()

    print("  ✓ Saved: cluster_analysis.png")

    # ========================================
    # Summary
    # ========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  High-error cases: {n_high_error:,}")
    print(f"  Clustered sample: {len(high_error):,}")
    print(f"  Optimal K: {optimal_k}")
    print(f"  Silhouette: {final_silhouette:.4f}")

    return {
        'high_error_df': high_error,
        'cluster_profiles': profiles_df,
        'optimal_k': optimal_k,
        'kmeans_model': kmeans_final,
        'silhouette_score': final_silhouette,
        'pca_model': pca,
        'scaler': scaler,
        'feature_cols': top_features,
        'threshold': threshold,
        'n_total_high_error': n_high_error
    }