# dml_analysis.py
"""
Double Machine Learning (DML) Pipeline for Causal Inference
============================================================

Uses existing preprocess_data_xgboost for data preparation.

Steps:
- Step 0: Create RAM-friendly 10% sample (3.5M rows)
- Step 1: Clean Outcome (Y Model) - Predict OCCSCORE ignoring immigration
- Step 2: Clean Treatment (T Model) - Predict is_immigrant from demographics
- Step 3: Compare Leftovers - Regress residuals to get causal effect
- Step 4: Cluster Filter - Analyze high-error cases
"""

import pandas as pd
from typing import Dict, List, Optional, Tuple
import os

from dml_pipeline.create_dml_sample import create_dml_sample
from dml_pipeline.dml_debug import debug_high_error_clustering
from dml_pipeline.dml_preprocess import preprocess_for_dml
from dml_pipeline.error_clusters import high_error_clustering
from dml_pipeline.metrics import plot_regression_diagnostics, plot_ate_results, export_ate_latex
from dml_pipeline.regression_effect import estimate_causal_effect
from dml_pipeline.train_outcome_model import train_outcome_model
from dml_pipeline.train_treatment_model import train_treatment_model


def run_dml_pipeline(
        data_path: str,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        exclude_cols: Optional[List[str]] = None,
        use_gpu: bool = True,
        run_clustering: bool = True,
        plot_metrics: bool = True,
        save_dir: str = './dml_results',
        verbose: bool = True
) -> Dict:
    """
    Run complete DML pipeline on preprocessed sample.

    Args:
        data_path: Path to sampled parquet (from Step 0)
        treatment_col: Treatment variable
        outcome_col: Outcome variable
        weight_col: Weight variable
        exclude_cols: Additional columns to exclude
        use_gpu: Use GPU if available
        run_clustering: Run Step 4 cluster analysis
        plot_metrics: Generate plots using evaluate_model_comprehensive
        save_dir: Directory to save plots and results
        verbose: Print progress

    Returns:
        Dictionary with all results
    """
    if verbose:
        print("\n" + "=" * 80)
        print("DOUBLE MACHINE LEARNING PIPELINE")
        print("=" * 80)

    # Create save directory
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Preprocess using existing function
    X_train, X_eval, y_train, y_eval, t_train, t_eval, w_train, w_eval = preprocess_for_dml(
        data_path=data_path,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        weight_col=weight_col,
        exclude_cols=exclude_cols,
        verbose=verbose
    )

    # Step 1: Outcome Model
    model_y, y_pred, res_y, metrics_y = train_outcome_model(
        X_train, y_train, X_eval, y_eval, w_train, w_eval,
        use_gpu=use_gpu,
        plot_metrics=plot_metrics,
        save_dir=save_dir,
        verbose=verbose
    )

    # Step 2: Treatment Model
    model_t, propensity, res_t, metrics_t = train_treatment_model(
        X_train, t_train, X_eval, t_eval, w_train, w_eval,
        use_gpu=use_gpu,
        plot_metrics=plot_metrics,
        save_dir=save_dir,
        verbose=verbose
    )

    # Step 3: Causal Effect
    causal = estimate_causal_effect(
        res_y, res_t,
        weights=w_eval.values if w_eval is not None else None,
        verbose=verbose
    )

    plot_ate_results(
        causal,
        treatment_name="Immigration",
        outcome_name="OCCSCORE",
        save_path="ate_forest_plot.png"
    )

    plot_regression_diagnostics(
        causal_results=causal,
        residuals_y=res_y,
        residuals_t=res_t,
        save_path=None,
        show_plot=True
    )

    # export_ate_latex(causal, "ate_table.tex")
    # Step 4: Clustering (optional) - Uses your perform_clustering!
    clusters = None
    if run_clustering:
        high_error_clustering(
            X_eval=X_eval,
            y_eval=y_eval,
            t_eval=t_eval,
            residuals_y=res_y,  # from Step 1
            residuals_t=res_t,  # from Step 2
            w_eval=w_eval,
            error_percentile=0.95,  # top 10% errors
            max_clusters=8,
            verbose=True
        )

    results = {
        # Causal effect (main result)
        'causal_effect': causal['causal_effect'],
        'standard_error': causal['standard_error'],
        't_statistic': causal['t_statistic'],
        'p_value': causal['p_value'],
        'ci_lower': causal['ci_lower'],
        'ci_upper': causal['ci_upper'],
        'significant': causal['significant_05'],
        'outcome_r2': metrics_y.get('test_r2', None),
        'treatment_accuracy': metrics_t.get('test_accuracy', None),
        'propensity_scores': propensity,
        'cluster_results': clusters,

        # Metadata

    }



    return results


# ============================================================
# CONVENIENCE: RUN FROM RAW FILE (Includes Step 0)
# ============================================================

def run_dml_from_file(
        data_path: str,
        sample_fraction: float = 0.1,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        exclude_cols: Optional[List[str]] = None,
        plot_metrics: bool = True,
        run_clustering: bool = True,
        save_dir: str = './dml_results',
        verbose: bool = True
) -> Dict:
    """
    Complete DML analysis from raw file, including sampling.

    This is the main entry point.

    Example:
        results = run_dml_from_file(
            "ipums_data_training_final.parquet",
            sample_fraction=0.1,
            plot_metrics=True,  # Uses your evaluate_model_comprehensive
            run_clustering=True  # Uses your perform_clustering
        )
        print(f"Causal Effect: {results['causal_effect']:.4f}")
    """
    if verbose:
        print("\n" + "=" * 80)
        print("DML ANALYSIS - FROM RAW FILE")
        print("=" * 80)

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Step 0: Create sample
    sample_path = create_dml_sample(
        data_path=data_path,
        sample_fraction=sample_fraction,
        save_sample=True,
        verbose=verbose
    )

    # Steps 1-4: Run pipeline
    results = run_dml_pipeline(
        data_path=sample_path,
        treatment_col=treatment_col,
        outcome_col=outcome_col,
        weight_col=weight_col,
        exclude_cols=exclude_cols,
        plot_metrics=plot_metrics,
        run_clustering=run_clustering,
        save_dir=save_dir,
        verbose=verbose
    )

    return results