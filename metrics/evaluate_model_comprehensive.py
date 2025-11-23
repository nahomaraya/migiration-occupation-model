import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
)
from typing import Dict, Any, Optional, Tuple, Union
import warnings
import matplotlib.pyplot as plt

from metrics.evaluate_classification import print_classification_metrics, plot_confusion_matrix, evaluate_classification
from metrics.evaluate_regression import evaluate_regression, print_regression_metrics, plot_regression_diagnostics
from metrics.feature_importance import plot_feature_importance

warnings.filterwarnings('ignore')

def plot_learning_curves(
        evals_result: Dict[str, Dict[str, list]],
        metric: str = 'rmse',
        title: Optional[str] = None,
        save_path: Optional[str] = None
) -> None:
    """
    Plot training/validation learning curves.

    Args:
        evals_result: XGBoost evals_result dictionary
        metric: Metric to plot
        title: Plot title
        save_path: Path to save figure
    """
    plt.figure(figsize=(10, 6))

    for dataset_name, metrics in evals_result.items():
        if metric in metrics:
            plt.plot(metrics[metric], label=f'{dataset_name.title()} {metric.upper()}', lw=2)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel(metric.upper(), fontsize=12)
    plt.title(title or f'Learning Curve - {metric.upper()}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved learning curves to: {save_path}")

    plt.show()


def evaluate_model_comprehensive(
        y_train_true: Union[pd.Series, np.ndarray],
        y_train_pred: Union[pd.Series, np.ndarray],
        y_test_true: Union[pd.Series, np.ndarray],
        y_test_pred: Union[pd.Series, np.ndarray],
        w_train: Optional[Union[pd.Series, np.ndarray]] = None,
        w_test: Optional[Union[pd.Series, np.ndarray]] = None,
        task: str = 'regression',
        y_train_pred_proba: Optional[np.ndarray] = None,
        y_test_pred_proba: Optional[np.ndarray] = None,
        class_names: Optional[list] = None,
        feature_importance: Optional[pd.DataFrame] = None,
        evals_result: Optional[Dict] = None,
        save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with all metrics and plots.

    Args:
        y_train_true, y_test_true: True values
        y_train_pred, y_test_pred: Predicted values
        w_train, w_test: Sample weights
        task: 'regression' or 'classification'
        y_train_pred_proba, y_test_pred_proba: Predicted probabilities (classification)
        class_names: Class names (classification)
        feature_importance: Feature importance DataFrame
        evals_result: XGBoost evals_result
        save_dir: Directory to save plots

    Returns:
        Dictionary with all metrics
    """
    results = {}

    if task == 'regression':
        # Calculate metrics
        train_metrics = evaluate_regression(y_train_true, y_train_pred, w_train, prefix='train_')
        test_metrics = evaluate_regression(y_test_true, y_test_pred, w_test, prefix='test_')
        results.update(train_metrics)
        results.update(test_metrics)

        # Print metrics
        print_regression_metrics(results, title="Regression Results")

        # Plot diagnostics
        plot_regression_diagnostics(
            y_test_true, y_test_pred, w_test,
            title="Test Set Diagnostics",
            save_path=f"{save_dir}/regression_diagnostics.png" if save_dir else None
        )

    elif task == 'classification':
        # Calculate metrics
        train_metrics = evaluate_classification(
            y_train_true, y_train_pred, y_train_pred_proba, w_train, prefix='train_'
        )
        test_metrics = evaluate_classification(
            y_test_true, y_test_pred, y_test_pred_proba, w_test, prefix='test_'
        )
        results.update(train_metrics)
        results.update(test_metrics)

        # Print metrics
        print_classification_metrics(results, title="Classification Results")

        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test_true, y_test_pred, target_names=class_names))

        # Plot confusion matrix
        plot_confusion_matrix(
            y_test_true, y_test_pred, class_names, w_test,
            title="Test Set Confusion Matrix",
            save_path=f"{save_dir}/confusion_matrix.png" if save_dir else None
        )

    # Feature importance
    if feature_importance is not None:
        plot_feature_importance(
            feature_importance,
            title="Top 20 Features",
            save_path=f"{save_dir}/feature_importance.png" if save_dir else None
        )

    # Learning curves
    if evals_result is not None:
        metric = 'rmse' if task == 'regression' else 'mlogloss'
        plot_learning_curves(
            evals_result,
            metric=metric,
            save_path=f"{save_dir}/learning_curves.png" if save_dir else None
        )

    return results