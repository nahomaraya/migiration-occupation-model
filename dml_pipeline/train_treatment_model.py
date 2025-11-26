import xgboost as xgb
import pandas as pd
import numpy as np
import gc
import os
import polars as pl
from typing import Dict, List, Optional, Tuple
from models.xgboost import check_gpu_availability
from metrics import evaluate_model_comprehensive

def train_treatment_model(
        X_train: pd.DataFrame,
        t_train: pd.Series,
        X_eval: pd.DataFrame,
        t_eval: pd.Series,
        w_train: Optional[pd.Series] = None,
        w_eval: Optional[pd.Series] = None,
        use_gpu: bool = True,
        num_boost_round: int = 500,
        plot_metrics: bool = True,
        save_dir: Optional[str] = None,
        verbose: bool = True
) -> Tuple[xgb.Booster, np.ndarray, np.ndarray, Dict]:
    """
    Step 2: Train the Treatment Model (T ~ W).

    Question: "Can I predict who is an immigrant from demographics?"

    Returns:
        (model, propensity_scores, residuals, metrics_dict)
        residuals = Actual (0/1) - Probability = "Pure immigration variation"
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 2: CLEAN THE TREATMENT (T MODEL)")
        print("=" * 80)
        print("\n  Question: Can we predict who is an immigrant?")
        print("  Goal: Remove selection bias → 'Pure' immigration variation")

    gpu_available = check_gpu_availability() if use_gpu else False

    params = {
        'device': 'cuda:0' if gpu_available else 'cpu',
        'tree_method': 'hist',
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'lambda': 1.0,
        'alpha': 0.1,
        'seed': 42
    }

    if verbose:
        print(f"\n  Training XGBoost Classifier on {params['device']}...")

    dtrain = xgb.DMatrix(X_train, label=t_train, weight=w_train)
    deval = xgb.DMatrix(X_eval, label=t_eval, weight=w_eval)

    # Train with evaluation tracking
    evals_result = {}
    model_t = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        early_stopping_rounds=500,
        evals=[(dtrain, 'train'), (deval, 'eval')],
        evals_result=evals_result,
        verbose_eval=False
    )

    # Propensity scores (predicted probabilities)
    t_train_pred_proba = model_t.predict(dtrain)
    propensity = model_t.predict(deval)

    # Binary predictions
    t_train_pred = (t_train_pred_proba > 0.5).astype(int)
    t_eval_pred = (propensity > 0.5).astype(int)

    # Residuals = Actual (0/1) - Predicted Probability
    residuals_t = t_eval.values - propensity

    # Get feature importance
    importance_dict = model_t.get_score(importance_type='gain')
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
            y_train_true=t_train,
            y_train_pred=t_train_pred,
            y_test_true=t_eval,
            y_test_pred=t_eval_pred,
            w_train=w_train,
            w_test=w_eval,
            task='classification',
            y_train_pred_proba=t_train_pred_proba,
            y_test_pred_proba=propensity,
            class_names=['Native', 'Immigrant'],
            feature_importance=feature_importance,
            evals_result=evals_result,
            save_dir=f"{save_dir}/treatment_model" if save_dir else None
        )
    else:
        # Just calculate basic metrics without plots
        accuracy = np.mean(t_eval_pred == t_eval.values)
        metrics = {'test_accuracy': accuracy}

        if verbose:
            print(f"\n  ✓ Model trained!")
            print(f"    Accuracy: {accuracy:.4f}")

    if verbose:
        print(f"\n  Result: Residual = Actual (0/1) - Probability")
        print(f"    This strips away selection bias")
        print(f"    (e.g., immigrants tend to be younger, live in cities)")

    del dtrain, deval
    gc.collect()

    return model_t, propensity, residuals_t, metrics

