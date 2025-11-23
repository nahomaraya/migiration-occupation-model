import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')


def plot_feature_importance(
        importance_df: pd.DataFrame,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_path: Optional[str] = None
) -> None:
    """
    Plot feature importance.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_n: Number of top features to show
        title: Plot title
        save_path: Path to save figure
    """
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance (Gain)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved feature importance to: {save_path}")

    plt.show()