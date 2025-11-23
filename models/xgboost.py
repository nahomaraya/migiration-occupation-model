# xgboost_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple


def train_xgboost_regressor(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        w_train: pd.Series,
        w_test: pd.Series,
        params: Dict[str, Any] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost regressor with GPU acceleration.

    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Target vectors
        w_train, w_test: Sample weights
        params: XGBoost parameters (optional)
        num_boost_round: Maximum boosting rounds
        early_stopping_rounds: Early stopping patience
        verbose: Print training progress

    Returns:
        Dictionary with model, predictions, and metrics
    """

    if verbose:
        print("=" * 80)
        print("TRAINING XGBOOST REGRESSOR")
        print("=" * 80)

    # Default parameters
    if params is None:
        params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.1,
            'min_child_weight': 5,
            'random_state': 42
        }

    if verbose:
        print("\nParameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test, weight=w_test)

    # Train
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    if verbose:
        print(f"\nTraining with {num_boost_round} max rounds, early stopping={early_stopping_rounds}...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50 if verbose else False
    )

    # Predictions
    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)

    # Metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train, sample_weight=w_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test, sample_weight=w_test))
    train_r2 = r2_score(y_train, y_pred_train, sample_weight=w_train)
    test_r2 = r2_score(y_test, y_pred_test, sample_weight=w_test)

    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print(f"Train RMSE: {train_rmse:.4f}")
        print(f"Test RMSE:  {test_rmse:.4f}")
        print(f"Train R²:   {train_r2:.4f}")
        print(f"Test R²:    {test_r2:.4f}")

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': importance.keys(),
        'importance': importance.values()
    }).sort_values('importance', ascending=False)

    return {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2
        },
        'feature_importance': importance_df,
        'evals_result': evals_result
    }


def train_xgboost_classifier(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        w_train: pd.Series,
        w_test: pd.Series,
        params: Dict[str, Any] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost classifier with GPU acceleration.
    """

    if verbose:
        print("=" * 80)
        print("TRAINING XGBOOST CLASSIFIER")
        print("=" * 80)

    # Remap classes to 0-indexed
    unique_classes = sorted(y_train.unique())
    num_classes = len(unique_classes)

    if min(unique_classes) != 0:
        class_mapping = {old: new for new, old in enumerate(unique_classes)}
        y_train_mapped = y_train.map(class_mapping)
        y_test_mapped = y_test.map(class_mapping)
    else:
        y_train_mapped = y_train
        y_test_mapped = y_test

    # Default parameters
    if params is None:
        params = {
            'device': 'cuda',
            'tree_method': 'hist',
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'lambda': 1.0,
            'alpha': 0.1,
            'min_child_weight': 5,
            'random_state': 42
        }

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_mapped, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test_mapped, weight=w_test)

    # Train
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50 if verbose else False
    )

    # Predictions
    y_pred_test_proba = model.predict(dtest)
    y_pred_test = np.argmax(y_pred_test_proba, axis=1)

    if verbose:
        print("\n" + "=" * 80)
        print("RESULTS")
        print("=" * 80)
        print("\nClassification Report:")
        print(classification_report(y_test_mapped, y_pred_test))

    return {
        'model': model,
        'y_pred_test': y_pred_test,
        'y_pred_test_proba': y_pred_test_proba,
        'evals_result': evals_result
    }