import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss
)
from typing import Dict, Any, Optional, Tuple, Union
import warnings

warnings.filterwarnings('ignore')

def evaluate_classification(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        y_pred_proba: Optional[np.ndarray] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        prefix: str = "",
        average: str = 'weighted'
) -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for AUC)
        sample_weight: Optional sample weights
        prefix: Prefix for metric names
        average: Averaging method for multiclass ('micro', 'macro', 'weighted')

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Accuracy
    metrics[f'{prefix}accuracy'] = accuracy_score(y_true, y_pred, sample_weight=sample_weight)

    # Precision, Recall, F1
    metrics[f'{prefix}precision'] = precision_score(
        y_true, y_pred, average=average, sample_weight=sample_weight, zero_division=0
    )
    metrics[f'{prefix}recall'] = recall_score(
        y_true, y_pred, average=average, sample_weight=sample_weight, zero_division=0
    )
    metrics[f'{prefix}f1'] = f1_score(
        y_true, y_pred, average=average, sample_weight=sample_weight, zero_division=0
    )

    # AUC (if probabilities provided)
    if y_pred_proba is not None:
        try:
            # For binary classification
            if y_pred_proba.ndim == 1 or y_pred_proba.shape[1] == 2:
                if y_pred_proba.ndim == 2:
                    y_pred_proba = y_pred_proba[:, 1]
                metrics[f'{prefix}auc'] = roc_auc_score(
                    y_true, y_pred_proba, sample_weight=sample_weight
                )
            # For multiclass
            else:
                metrics[f'{prefix}auc'] = roc_auc_score(
                    y_true, y_pred_proba,
                    average=average,
                    multi_class='ovr',
                    sample_weight=sample_weight
                )
        except Exception as e:
            print(f"Warning: Could not calculate AUC - {e}")

    # Log loss (if probabilities provided)
    if y_pred_proba is not None:
        try:
            metrics[f'{prefix}logloss'] = log_loss(
                y_true, y_pred_proba, sample_weight=sample_weight
            )
        except:
            pass

    return metrics


def print_classification_metrics(
        metrics: Dict[str, float],
        title: str = "Classification Metrics"
) -> None:
    """
    Pretty print classification metrics.

    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)

    # Group by prefix
    train_metrics = {k: v for k, v in metrics.items() if k.startswith('train_')}
    test_metrics = {k: v for k, v in metrics.items() if k.startswith('test_')}

    if train_metrics and test_metrics:
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
        for key, value in metrics.items():
            metric_name = key.replace('train_', '').replace('test_', '').upper()
            print(f"{metric_name:<20} {value:.4f}")


def plot_confusion_matrix(
        y_true: Union[pd.Series, np.ndarray],
        y_pred: Union[pd.Series, np.ndarray],
        class_names: Optional[list] = None,
        sample_weight: Optional[Union[pd.Series, np.ndarray]] = None,
        normalize: bool = False,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        sample_weight: Optional sample weights
        normalize: Whether to normalize
        title: Plot title
        save_path: Path to save figure
    """
    cm = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = '.0f'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap='Blues',
        xticklabels=class_names if class_names else 'auto',
        yticklabels=class_names if class_names else 'auto',
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    plt.ylabel('Actual', fontsize=12)
    plt.xlabel('Predicted', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to: {save_path}")

    plt.show()


def plot_roc_curve(
        y_true: Union[pd.Series, np.ndarray],
        y_pred_proba: np.ndarray,
        title: str = "ROC Curve",
        save_path: Optional[str] = None
) -> None:
    """
    Plot ROC curve (binary classification only).

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved ROC curve to: {save_path}")

    plt.show()
