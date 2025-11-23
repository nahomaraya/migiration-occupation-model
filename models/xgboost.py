# xgboost_model.py
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple, Optional
import subprocess
import sys

from metrics.evaluate_model_comprehensive import evaluate_model_comprehensive


def check_gpu_availability() -> Tuple[bool, str]:
    try:
        # Check CUDA availability
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return True, f"CUDA GPU detected: {gpu_name} ({gpu_memory:.1f}GB)"
        else:
            return False, "CUDA not available"
    except ImportError:
        # Fallback: Try nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                gpu_info = result.stdout.strip()
                return True, f"GPU detected: {gpu_info}"
            else:
                return False, "No NVIDIA GPU detected"
        except FileNotFoundError:
            return False, "nvidia-smi not found - GPU detection failed"


def get_optimal_tree_method(force_gpu: bool = True) -> str:
    has_gpu, gpu_info = check_gpu_availability()

    if has_gpu or force_gpu:
        print(f"GPU Mode: {gpu_info}")
        return 'hist'  # GPU-compatible method
    else:
        print(f"CPU Mode: {gpu_info}")
        return 'hist'  # Still works on CPU


def train_xgboost_regressor(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        w_train: pd.Series,
        w_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        use_gpu: bool = True,
        gpu_id: int = 0,
        verbose: bool = True,
        save_dir: Optional[str] = None
) -> Dict[str, Any]:
    if verbose:
        print("=" * 80)
        print("TRAINING XGBOOST REGRESSOR")
        print("=" * 80)

    # Check GPU availability
    has_gpu, gpu_info = check_gpu_availability()
    if verbose:
        print(f"\n{gpu_info}")
        if use_gpu and not has_gpu:
            print("GPU requested but not available, falling back to CPU")
            use_gpu = False

    # Default parameters with GPU optimization
    if params is None:
        params = {
            # GPU Settings
            'device': f'cuda:{gpu_id}' if use_gpu else 'cpu',
            'tree_method': 'hist',  # Required for GPU

            # Objective & Metrics
            'objective': 'reg:squarederror',
            'eval_metric': ['rmse', 'mae'],

            # Model Complexity
            'max_depth': 8,
            'eta': 0.05,  # Learning rate
            'subsample': 0.8,
            'colsample_bytree': 0.8,

            # Regularization
            'lambda': 1.0,  # L2 regularization
            'alpha': 0.1,  # L1 regularization
            'min_child_weight': 10,

            # GPU-specific optimizations
            'max_bin': 256 if use_gpu else 256,  # Histogram bins (higher = more memory)
            'grow_policy': 'depthwise',  # Better for GPU

            # Other
            'random_state': 42,
            'nthread': -1  # Use all CPU threads for data loading
        }
    else:
        # Ensure GPU is set if requested
        if use_gpu:
            params['device'] = f'cuda:{gpu_id}'
            params['tree_method'] = 'hist'

    if verbose:
        print("\nTraining Parameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

        print(f"\nDataset Info:")
        print(f"  Training samples: {X_train.shape[0]:,}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Memory (train): {X_train.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # Create DMatrix with GPU support
    if verbose:
        print("\nCreating DMatrix objects...")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=w_train,
        enable_categorical=True  # Set True if you have categorical features
    )
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        weight=w_test,
        enable_categorical=True
    )

    # Train with GPU
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    if verbose:
        print(f"\nStarting training...")
        print(f"  Max rounds: {num_boost_round}")
        print(f"  Early stopping: {early_stopping_rounds} rounds")
        print(f"  Device: {params['device']}")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50 if verbose else False
    )

    if verbose:
        print(f"\n✓ Training complete!")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Best score: {model.best_score:.4f}")

    # Predictions
    if verbose:
        print("\nGenerating predictions...")

    y_pred_train = model.predict(dtrain)
    y_pred_test = model.predict(dtest)

    # Metrics

    # Feature importance
    importance = model.get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': importance.keys(),
        'importance': importance.values()
    }).sort_values('importance', ascending=False)

    results = evaluate_model_comprehensive(
        y_train_true=y_train,
        y_train_pred=y_pred_train,
        y_test_true=y_test,
        y_test_pred=y_pred_test,
        w_train=w_train,
        w_test=w_test,
        task='regression',
        feature_importance=importance_df,
        evals_result=evals_result,
        save_dir=save_dir
    )



    if verbose:
        print("\nTop 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    return {
        'model': model,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'metrics': results,
        'feature_importance': importance_df,
        'evals_result': evals_result,
        'best_iteration': model.best_iteration,
        'gpu_used': use_gpu
    }


def train_xgboost_classifier(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        w_train: pd.Series,
        w_test: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        use_gpu: bool = True,
        gpu_id: int = 0,
        verbose: bool = True
) -> Dict[str, Any]:
    """
    Train XGBoost classifier with GPU acceleration.
    """

    if verbose:
        print("=" * 80)
        print("TRAINING XGBOOST CLASSIFIER")
        print("=" * 80)

    # Check GPU
    has_gpu, gpu_info = check_gpu_availability()
    if verbose:
        print(f"\n{gpu_info}")
        if use_gpu and not has_gpu:
            print("⚠️  GPU requested but not available, falling back to CPU")
            use_gpu = False

    # Remap classes to 0-indexed
    unique_classes = sorted(y_train.unique())
    num_classes = len(unique_classes)

    if verbose:
        print(f"\nClass Distribution:")
        print(f"  Unique classes: {num_classes}")
        print(f"  Classes: {unique_classes}")

    if min(unique_classes) != 0:
        class_mapping = {old: new for new, old in enumerate(unique_classes)}
        y_train_mapped = y_train.map(class_mapping)
        y_test_mapped = y_test.map(class_mapping)
        if verbose:
            print(f"  Remapped to 0-indexed: {list(class_mapping.values())}")
    else:
        y_train_mapped = y_train
        y_test_mapped = y_test

    # Default parameters
    if params is None:
        params = {
            # GPU Settings
            'device': f'cuda:{gpu_id}' if use_gpu else 'cpu',
            'tree_method': 'hist',

            # Objective & Metrics
            'objective': 'multi:softprob',
            'num_class': num_classes,
            'eval_metric': 'mlogloss',

            # Model Complexity
            'max_depth': 8,
            'eta': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,

            # Regularization
            'lambda': 1.0,
            'alpha': 0.1,
            'min_child_weight': 5,

            # GPU-specific
            'max_bin': 256,
            'grow_policy': 'depthwise',

            # Other
            'random_state': 42,
            'nthread': -1
        }
    else:
        if use_gpu:
            params['device'] = f'cuda:{gpu_id}'
            params['tree_method'] = 'hist'
        params['num_class'] = num_classes

    if verbose:
        print("\nTraining Parameters:")
        for key, val in params.items():
            print(f"  {key}: {val}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train_mapped, weight=w_train)
    dtest = xgb.DMatrix(X_test, label=y_test_mapped, weight=w_test)

    # Train
    evals = [(dtrain, 'train'), (dtest, 'test')]
    evals_result = {}

    if verbose:
        print(f"\nStarting training on {params['device']}...")

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=50 if verbose else False
    )

    if verbose:
        print(f"\n✓ Training complete!")
        print(f"  Best iteration: {model.best_iteration}")

    # Predictions
    y_pred_train_proba = model.predict(dtrain)
    y_pred_test_proba = model.predict(dtest)

    y_pred_train = np.argmax(y_pred_train_proba, axis=1)
    y_pred_test = np.argmax(y_pred_test_proba, axis=1)

    if verbose:
        print("\n" + "=" * 80)
        print("CLASSIFICATION RESULTS")
        print("=" * 80)
        print("\nTest Set Classification Report:")
        print(classification_report(
            y_test_mapped,
            y_pred_test,
            target_names=[f"Class_{i}" for i in range(num_classes)]
        ))

        # Confusion matrix
        cm = confusion_matrix(y_test_mapped, y_pred_test, sample_weight=w_test)
        print("\nWeighted Confusion Matrix:")
        print(cm)

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
        'y_pred_train_proba': y_pred_train_proba,
        'y_pred_test_proba': y_pred_test_proba,
        'evals_result': evals_result,
        'feature_importance': importance_df,
        'best_iteration': model.best_iteration,
        'class_mapping': class_mapping if min(unique_classes) != 0 else None,
        'gpu_used': use_gpu
    }