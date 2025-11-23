import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist


def perform_clustering(x_train, x_val, y_train=None, y_val=None, max_clusters=10):
    """
    Clustering analysis with K-Means and Hierarchical Clustering.
    """
    # Calculate feature importance based on variance
    feature_variance = x_train.var().sort_values(ascending=False)
    print("\nTop 10 Features by Variance:")
    print(feature_variance.head(10))

    # Calculate feature correlations
    corr_matrix = x_train.corr()

    # """
    #   Feature Selection Rationale:")
    #  -All features have been standardized (mean=0, std=1)")
    #   Features with constant values have been removed")
    #   Highly correlated features (>0.95) have been removed")
    #   Remaining features capture diverse aspects of the data")
    #  Total features used for clustering: {x_train.shape[1]}")
    # """
    print("\nDETERMINING OPTIMAL NUMBER OF CLUSTERS")
    inertias = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(x_train)
        inertias.append(kmeans.inertia_)
        print(f"  K={k}: Inertia={kmeans.inertia_:.2f}")

    # Plot Elbow Method
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=12)
    plt.title('Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Silhouette Score Method
    print("Silhouette Score Method:")
    silhouette_scores = []

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(x_train)
        score = silhouette_score(x_train, labels)
        silhouette_scores.append(score)
        print(f"  K={k}: Silhouette Score={score:.4f}")

    # Plot Silhouette Scores
    plt.subplot(1, 2, 2)
    plt.plot(K_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Score for Optimal K', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Determine optimal K
    optimal_k = K_range[np.argmax(silhouette_scores)]
    print(f"\n Optimal number of clusters: K={optimal_k} (highest silhouette score)")


    print("\n BUILDING K-MEANS CLUSTERING MODEL")


    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    kmeans_labels_train = kmeans.fit_predict(x_train)
    kmeans_labels_val = kmeans.predict(x_val)

    print(f"K-Means model built with {optimal_k} clusters")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

    print("\n BUILDING HIERARCHICAL CLUSTERING MODEL")

    # Create dendrogram to visualize hierarchy
    print("Generating dendrogram...")
    plt.figure(figsize=(15, 7))

    # Use a sample if dataset is too large
    sample_size = min(100, len(x_train))
    sample_indices = np.random.choice(len(x_train), sample_size, replace=False)
    x_sample = x_train.iloc[sample_indices]

    linkage_matrix = linkage(x_sample, method='ward')
    dendrogram(linkage_matrix, truncate_mode='level', p=5)
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Hierarchical Clustering Dendrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Build hierarchical clustering with same number of clusters
    hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
    hierarchical_labels_train = hierarchical.fit_predict(x_train)
    hierarchical_labels_val = hierarchical.fit_predict(x_val)

    print(f"Hierarchical clustering model built with {optimal_k} clusters")

    print("\n MODEL EVALUATION - INTERNAL METRICS")

    # K-Means Evaluation
    print("\n K-MEANS CLUSTERING:")
    kmeans_silhouette = silhouette_score(x_train, kmeans_labels_train)
    kmeans_davies_bouldin = davies_bouldin_score(x_train, kmeans_labels_train)
    kmeans_calinski = calinski_harabasz_score(x_train, kmeans_labels_train)

    print(f"  Silhouette Score: {kmeans_silhouette:.4f} (higher is better, range: -1 to 1)")
    print(f"  Davies-Bouldin Index: {kmeans_davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Index: {kmeans_calinski:.2f} (higher is better)")

    # Hierarchical Clustering Evaluation
    print("\n HIERARCHICAL CLUSTERING:")
    hierarchical_silhouette = silhouette_score(x_train, hierarchical_labels_train)
    hierarchical_davies_bouldin = davies_bouldin_score(x_train, hierarchical_labels_train)
    hierarchical_calinski = calinski_harabasz_score(x_train, hierarchical_labels_train)

    print(f"  Silhouette Score: {hierarchical_silhouette:.4f} (higher is better)")
    print(f"  Davies-Bouldin Index: {hierarchical_davies_bouldin:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Index: {hierarchical_calinski:.2f} (higher is better)")

    # Cluster size distribution
    print("\n CLUSTER SIZE DISTRIBUTION:")
    print("\nK-Means:")
    kmeans_cluster_counts = pd.Series(kmeans_labels_train).value_counts().sort_index()
    for cluster, count in kmeans_cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples ({count / len(kmeans_labels_train) * 100:.1f}%)")

    print("\nHierarchical:")
    hierarchical_cluster_counts = pd.Series(hierarchical_labels_train).value_counts().sort_index()
    for cluster, count in hierarchical_cluster_counts.items():
        print(f"  Cluster {cluster}: {count} samples ({count / len(hierarchical_labels_train) * 100:.1f}%)")


    print("\n" + "=" * 80)
    print("\n CLUSTER CHARACTERISTICS (K-MEANS)")

    # Add cluster labels to training data
    x_train_clustered = x_train.copy()
    x_train_clustered['Cluster'] = kmeans_labels_train

    # Analyze each cluster
    for cluster in range(optimal_k):
        print(f"\n CLUSTER {cluster}:")
        cluster_data = x_train_clustered[x_train_clustered['Cluster'] == cluster]
        cluster_size = len(cluster_data)
        print(f"   Size: {cluster_size} samples ({cluster_size / len(x_train) * 100:.1f}%)")

        # Calculate mean values for each feature in this cluster
        cluster_means = cluster_data.drop('Cluster', axis=1).mean()

        # Find most distinguishing features (furthest from 0 since data is standardized)
        feature_importance = cluster_means.abs().sort_values(ascending=False)

        print(f"\n   Top 5 Distinguishing Features:")
        for i, (feature, value) in enumerate(feature_importance.head(5).items(), 1):
            actual_value = cluster_means[feature]
            direction = "HIGH" if actual_value > 0 else "LOW"
            print(f"   {i}. {feature}: {actual_value:.3f} ({direction})")

    print("\n VISUALIZING FEATURE CONTRIBUTIONS")
    print("-" * 80)

    # Heatmap of cluster centers
    plt.figure(figsize=(14, 8))

    # Get top 15 most variable features across clusters
    cluster_centers_df = pd.DataFrame(
        kmeans.cluster_centers_,
        columns=x_train.columns,
        index=[f'Cluster {i}' for i in range(optimal_k)]
    )

    feature_variance_across_clusters = cluster_centers_df.var().sort_values(ascending=False)
    top_features = feature_variance_across_clusters.head(15).index

    sns.heatmap(
        cluster_centers_df[top_features].T,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Standardized Value'}
    )
    plt.title('Cluster Centers - Top 15 Distinguishing Features', fontsize=14, fontweight='bold')
    plt.xlabel('Cluster', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Bar plot of cluster characteristics
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.ravel()

    for i, feature in enumerate(top_features[:6]):
        cluster_means = []
        cluster_labels = []
        for cluster in range(optimal_k):
            cluster_data = x_train_clustered[x_train_clustered['Cluster'] == cluster]
            cluster_means.append(cluster_data[feature].mean())
            cluster_labels.append(f'C{cluster}')

        axes[i].bar(cluster_labels, cluster_means, color=plt.cm.Set3(range(optimal_k)))
        axes[i].set_title(f'{feature}', fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Mean Value (standardized)', fontsize=10)
        axes[i].axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
        axes[i].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Feature Distributions Across Clusters', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()

    # ========================================================================
    # 3.3: 2D and 3D Cluster Visualization
    # ========================================================================
    print("\n CLUSTER VISUALIZATION (PCA PROJECTION)")
    print("-" * 80)

    # PCA for 2D visualization
    pca_2d = PCA(n_components=2, random_state=42)
    x_train_pca_2d = pca_2d.fit_transform(x_train)

    print(f"PCA explained variance (2D): {pca_2d.explained_variance_ratio_.sum():.2%}")

    # 2D Scatter Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # K-Means
    scatter1 = axes[0].scatter(
        x_train_pca_2d[:, 0],
        x_train_pca_2d[:, 1],
        c=kmeans_labels_train,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=50
    )
    axes[0].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    axes[0].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    axes[0].set_title('K-Means Clustering (2D PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=axes[0], label='Cluster')
    axes[0].grid(True, alpha=0.3)

    # Hierarchical
    scatter2 = axes[1].scatter(
        x_train_pca_2d[:, 0],
        x_train_pca_2d[:, 1],
        c=hierarchical_labels_train,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=50
    )
    axes[1].set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    axes[1].set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    axes[1].set_title('Hierarchical Clustering (2D PCA)', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=axes[1], label='Cluster')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # PCA for 3D visualization
    pca_3d = PCA(n_components=3, random_state=42)
    x_train_pca_3d = pca_3d.fit_transform(x_train)

    print(f"PCA explained variance (3D): {pca_3d.explained_variance_ratio_.sum():.2%}")

    # 3D Scatter Plot
    fig = plt.figure(figsize=(16, 7))

    # K-Means 3D
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(
        x_train_pca_3d[:, 0],
        x_train_pca_3d[:, 1],
        x_train_pca_3d[:, 2],
        c=kmeans_labels_train,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=30
    )
    ax1.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax1.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax1.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=10)
    ax1.set_title('K-Means Clustering (3D PCA)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, shrink=0.5, label='Cluster')

    # Hierarchical 3D
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(
        x_train_pca_3d[:, 0],
        x_train_pca_3d[:, 1],
        x_train_pca_3d[:, 2],
        c=hierarchical_labels_train,
        cmap='viridis',
        alpha=0.6,
        edgecolors='black',
        linewidth=0.5,
        s=30
    )
    ax2.set_xlabel(f'PC1 ({pca_3d.explained_variance_ratio_[0]:.1%})', fontsize=10)
    ax2.set_ylabel(f'PC2 ({pca_3d.explained_variance_ratio_[1]:.1%})', fontsize=10)
    ax2.set_zlabel(f'PC3 ({pca_3d.explained_variance_ratio_[2]:.1%})', fontsize=10)
    ax2.set_title('Hierarchical Clustering (3D PCA)', fontsize=12, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, shrink=0.5, label='Cluster')

    plt.tight_layout()
    plt.show()


    print("\n CLUSTER SEPARATION ANALYSIS")

    # Calculate pairwise distances between cluster centers
    cluster_center_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_, metric='euclidean')

    print("\nPairwise Distances Between Cluster Centers:")
    distance_df = pd.DataFrame(
        cluster_center_distances,
        index=[f'C{i}' for i in range(optimal_k)],
        columns=[f'C{i}' for i in range(optimal_k)]
    )
    print(distance_df.round(3))

    # Visualize cluster separation
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        distance_df,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        square=True,
        cbar_kws={'label': 'Euclidean Distance'}
    )
    plt.title('Pairwise Distances Between Cluster Centers', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Silhouette analysis per cluster
    from sklearn.metrics import silhouette_samples

    silhouette_vals = silhouette_samples(x_train, kmeans_labels_train)

    plt.figure(figsize=(12, 6))
    y_lower = 10

    for i in range(optimal_k):
        cluster_silhouette_vals = silhouette_vals[kmeans_labels_train == i]
        cluster_silhouette_vals.sort()

        size_cluster_i = cluster_silhouette_vals.shape[0]
        y_upper = y_lower + size_cluster_i

        color = plt.cm.viridis(float(i) / optimal_k)
        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_silhouette_vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster_i, f'C{i}')
        y_lower = y_upper + 10

    plt.axvline(x=kmeans_silhouette, color="red", linestyle="--", linewidth=2,
                label=f'Average: {kmeans_silhouette:.3f}')
    plt.xlabel('Silhouette Coefficient', fontsize=12)
    plt.ylabel('Cluster', fontsize=12)
    plt.title('Silhouette Analysis - Cluster Quality', fontsize=14, fontweight='bold')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nClustering analysis complete!")
    # print("\nSUMMARY:")
    # print(f"  - Optimal clusters: {optimal_k}")
    # print(f"  - K-Means Silhouette: {kmeans_silhouette:.4f}")
    # print(f"  - Hierarchical Silhouette: {hierarchical_silhouette:.4f}")
    # print(f"  - Davies-Bouldin (K-Means): {kmeans_davies_bouldin:.4f}")
    # print(
    #     f"  - Clusters are {'well-separated' if kmeans_silhouette > 0.5 else 'moderately separated' if kmeans_silhouette > 0.3 else 'poorly separated'}")

    return {
        'kmeans_model': kmeans,
        'hierarchical_model': hierarchical,
        'kmeans_labels_train': kmeans_labels_train,
        'kmeans_labels_val': kmeans_labels_val,
        'hierarchical_labels_train': hierarchical_labels_train,
        'hierarchical_labels_val': hierarchical_labels_val,
        'optimal_k': optimal_k,
        'metrics': {
            'kmeans': {
                'silhouette': kmeans_silhouette,
                'davies_bouldin': kmeans_davies_bouldin,
                'calinski_harabasz': kmeans_calinski
            },
            'hierarchical': {
                'silhouette': hierarchical_silhouette,
                'davies_bouldin': hierarchical_davies_bouldin,
                'calinski_harabasz': hierarchical_calinski
            }
        }
    }