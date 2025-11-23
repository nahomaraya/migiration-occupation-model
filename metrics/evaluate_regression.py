import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    # Regression metrics
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
from typing import Dict, Any, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

def evaluate_regression(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        prefix: str = ""
) -> Dict[str, float]:
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values
        sample_weight: Optional sample weights
        prefix: Prefix for metric names (e.g., "train_" or "test_")

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # RMSE
    mse = mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    metrics[f'{prefix}rmse'] = np.sqrt(mse)
    metrics[f'{prefix}mse'] = mse

    # MAE
    metrics[f'{prefix}mae'] = mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

    # R²
    metrics[f'{prefix}r2'] = r2_score(y_true, y_pred, sample_weight=sample_weight)

    # MAPE (if no zeros in y_true)
    if not np.any(y_true == 0):
        try:
            metrics[f'{prefix}mape'] = mean_absolute_percentage_error(y_true, y_pred, sample_weight=sample_weight)
        except:
            pass

    # Residuals statistics
    residuals = y_true - y_pred
    if sample_weight is not None:
        residuals_weighted = residuals * sample_weight
        metrics[f'{prefix}residual_mean'] = np.average(residuals, weights=sample_weight)
        metrics[f'{prefix}residual_std'] = np.sqrt(np.average(residuals ** 2, weights=sample_weight))
    else:
        metrics[f'{prefix}residual_mean'] = residuals.mean()
        metrics[f'{prefix}residual_std'] = residuals.std()

    return metrics

def print_regression_metrics(
        metrics: Dict[str, float],
        title: str = "Regression Metrics"
) -> None:
    """
    Pretty print regression metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

    # Group by prefix (train/test)
    train_metrics = {k: v for k, v in metrics.items() if k.startswith('train_')}
    test_metrics = {k: v for k, v in metrics.items() if k.startswith('test_')}

    if train_metrics and test_metrics:
        # Side by side comparison
        print(f"{'Metric':<20} {'Train':<15} {'Test':<15}")
        print("-" * 50)

        for key in train_metrics.keys():
            metric_name = key.replace('train_', '').upper()
            train_val = train_metrics[key]
            test_key = key.replace('train_', 'test_')
            test_val = test_metrics.get(test_key, None)

            if test_val is not None:
                print(f"{metric_name:<20} {train_val:<15.4f} {test_val:<15.4f}")
            else:
                print(f"{metric_name:<20} {train_val:<15.4f}")
    else:
        # Single set of metrics
        for key, value in metrics.items():
            metric_name = key.replace('train_', '').replace('test_', '').upper()
            print(f"{metric_name:<20} {value:.4f}")


def plot_regression_diagnostics(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        title: str = "Regression Diagnostics",
        save_path: Optional[str] = None,
        max_points: int = 10000
) -> None:
    """
    Create diagnostic plots for regression models.

    Args:
        y_true: True values
        y_pred: Predicted values
        sample_weight: Optional sample weights
        title: Plot title
        save_path: Path to save figure (optional)
        max_points: Maximum points to plot (for large datasets)
    """
    # Subsample if too many points
    if len(y_true) > max_points:
        idx = np.random.choice(len(y_true), max_points, replace=False)
        y_true_plot = y_true.iloc[idx] if isinstance(y_true, pd.Series) else y_true[idx]
        y_pred_plot = y_pred[idx]
        weights_plot = sample_weight.iloc[idx] if sample_weight is not None and isinstance(sample_weight,
                                                                                           pd.Series) else (
            sample_weight[idx] if sample_weight is not None else None)
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
        weights_plot = sample_weight

    residuals = y_true_plot - y_pred_plot

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1. Actual vs Predicted
    ax = axes[0, 0]
    ax.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=1)

    # Perfect prediction line
    min_val = min(y_true_plot.min(), y_pred_plot.min())
    max_val = max(y_true_plot.max(), y_pred_plot.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Residuals vs Predicted
    ax = axes[0, 1]
    ax.scatter(y_pred_plot, residuals, alpha=0.3, s=1)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Residuals', fontsize=12)
    ax.set_title('Residuals vs Predicted', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 3. Residual Distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Residuals', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Q-Q Plot
    ax = axes[1, 1]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved diagnostics to: {save_path}")

    plt.show()
