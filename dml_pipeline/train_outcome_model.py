
import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import os
import polars as pl
from typing import Dict, List, Optional, Tuple
from models.xgboost import check_gpu_availability
from metrics import evaluate_model_comprehensive
def train_outcome_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_eval: pd.DataFrame,
        y_eval: pd.Series,
        w_train: Optional[pd.Series] = None,
        w_eval: Optional[pd.Series] = None,
        use_gpu: bool = True,
        num_boost_round: int = 500,
        plot_metrics: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = True
) -> Tuple[xgb.Booster, np.ndarray, np.ndarray, Dict]:
    """
    Step 1: Train the Outcome Model (Y ~ W).

    Question: "Ignoring immigration status, can I predict OCCSCORE
    from Age, Education, Region, etc.?"

    Returns:
        (model, predictions, residuals, metrics_dict)
        residuals = Actual - Predicted = "Unexplained success"
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 1: CLEAN THE OUTCOME (Y MODEL)")
        print("=" * 80)
        print("\n  Question: Can we predict OCCSCORE ignoring immigration?")
        print("  Goal: Residuals = 'Unexplained Success'")

    # Check GPU
    gpu_available = check_gpu_availability() if use_gpu else False

    params = {
        'device': 'cuda:0' if gpu_available else 'cpu',
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'alpha': 0.1,
        'seed': 42
    }

    if verbose:
        print(f"\n  Training XGBoost Regressor on {params['device']}...")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train, enable_categorical=True)
    deval = xgb.DMatrix(X_eval, label=y_eval, weight=w_eval, enable_categorical=True)

    # Train with evaluation tracking
    evals_result = {}
    model_y = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=500,
        evals=[(dtrain, 'train'), (deval, 'eval')],
        evals_result=evals_result,

        verbose_eval=False
    )

    # Predict on both sets
    y_train_pred = model_y.predict(dtrain)
    y_pred = model_y.predict(deval)

    # Residuals = Actual - Predicted
    residuals_y = y_eval.values - y_pred

    # Get feature importance
    importance_dict = model_y.get_score(importance_type='gain')
    feature_importance = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)

    # Use your evaluate_model_comprehensive for metrics and plots!
    metrics = {}
    if plot_metrics:
        if verbose:
            print("\n  Evaluating model with comprehensive metrics...")

        metrics = evaluate_model_comprehensive(
            y_train_true=y_train,
            y_train_pred=y_train_pred,
            y_test_true=y_eval,
            y_test_pred=y_pred,
            w_train=w_train,
            w_test=w_eval,
            task='regression',
            feature_importance=feature_importance,
            evals_result=evals_result,
            save_dir=f"{save_dir}/outcome_model" if save_dir else None
        )
    else:
        # Just calculate basic metrics without plots
        rmse = np.sqrt(np.mean(residuals_y ** 2))
        r2 = 1 - (np.sum(residuals_y ** 2) / np.sum((y_eval - y_eval.mean()) ** 2))
        metrics = {'test_rmse': rmse, 'test_r2': r2}

        if verbose:
            print(f"\n  ✓ Model trained!")
            print(f"    RMSE: {rmse:.4f}")
            print(f"    R²: {r2:.4f}")

    if verbose:
        print(f"\n  Result: Residual = Actual - Predicted")
        print(f"    Positive residual → doing BETTER than predicted")
        print(f"    Negative residual → doing WORSE than predicted")

    del dtrain, deval
    gc.collect()

    return model_y, y_pred, residuals_y, metrics
