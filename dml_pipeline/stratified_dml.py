# stratified_dml_simple.py
"""
Simplified Stratified DML Analysis
==================================

Direct integration with your existing pipeline.
Minimal dependencies, maximum clarity.

Usage:
    python stratified_dml_simple.py
"""

import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import gc


# ============================================================
# ADJUST THIS IMPORT TO MATCH YOUR PROJECT
# ============================================================
# Option 1: If your DML is a function
# from dml_pipeline.run_dml_pipeline import run_dml_pipeline

# Option 2: If you run DML as a script, we'll define a wrapper
# that calls your existing code


def run_dml_on_subset(
        df: pd.DataFrame,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        weight_col: str = 'perwt',
        exclude_cols: List[str] = None,
        save_dir: str = './dml_temp',
        verbose: bool = False
) -> Dict:
    """
    Run your existing DML pipeline on a data subset.

    Since your pipeline expects a file path, we:
    1. Save the subset to a temporary parquet file
    2. Call your run_dml_pipeline with that path
    3. Clean up the temp file
    """
    import os
    import tempfile

    from dml_pipeline.run_dml_pipeline import run_dml_pipeline

    # Create temp directory if needed
    os.makedirs(save_dir, exist_ok=True)

    # Save subset to temporary parquet file
    temp_path = os.path.join(save_dir, f"temp_stratum_{id(df)}.parquet")

    try:
        # Save to parquet
        df.to_parquet(temp_path, index=False)

        # Call your existing pipeline
        results = run_dml_pipeline(
            data_path=temp_path,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            weight_col=weight_col,
            exclude_cols=exclude_cols,
            use_gpu=True,
            run_clustering=False,  # Skip clustering for stratified runs
            plot_metrics=False,  # Skip plots for stratified runs
            save_dir=save_dir,
            verbose=verbose
        )

        # Extract results (matching your return format)
        return {
            'effect': results.get('causal_effect'),
            'se': results.get('standard_error'),
            't_stat': results.get('t_statistic'),
            'p_value': results.get('p_value'),
            'ci_lower': results.get('ci_lower'),
            'ci_upper': results.get('ci_upper'),
            'significant': results.get('significant', False),
            'outcome_r2': results.get('outcome_r2'),
            'treatment_accuracy': results.get('treatment_accuracy'),
        }

    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ============================================================
# STRATIFICATION DEFINITIONS (PANDAS-COMPATIBLE)
# ============================================================

YEARS_IN_US_STRATA = {
    "0-5 years": lambda df: df[(df['years_in_us'] >= 0) & (df['years_in_us'] <= 5)],
    "5-10 years": lambda df: df[(df['years_in_us'] > 5) & (df['years_in_us'] <= 10)],
    "10-20 years": lambda df: df[(df['years_in_us'] > 10) & (df['years_in_us'] <= 20)],
    "20+ years": lambda df: df[df['years_in_us'] > 20],
}

EDUCATION_STRATA = {
    "STEM": lambda df: df[df['is_stem'] == 1],
    "Non-STEM": lambda df: df[df['is_stem'] == 0],
}

ORIGIN_REGION_STRATA = {
    "High_Income_Western": lambda df: df[df['origin_development_level'] == "High_Income_Western"],
    "MENA_Africa": lambda df: df[df['origin_development_level'] == "MENA_Africa"],
    "Latin_America": lambda df: df[df['origin_development_level'] == "Latin_America"],
    "Developing_Asia": lambda df: df[df['origin_development_level'] == "Developing_Asia"],
    "Upper_Middle_Europe": lambda df: df[df['origin_development_level'] == "Upper_Middle_Europe"],
    "High_Income_Asia": lambda df: df[df['origin_development_level'] == "High_Income_Asia"],
}

# COHORT_STRATA = {
#     "Pre-2000": lambda df: df[df['yrimmig'] < 2000],
#     "2000-2008": lambda df: df[(df['yrimmig'] >= 2000) & (df['yrimmig'] < 2008)],
#     "2008-2016": lambda df: df[(df['yrimmig'] >= 2008) & (df['yrimmig'] < 2016)],
#     "2016-2020": lambda df: df[(df['yrimmig'] >= 2016) & (df['yrimmig'] < 2020)],
#     "2020+": lambda df: df[df['yrimmig'] >= 2020],
# }

# Columns to always exclude from DML features
STANDARD_EXCLUDE = [
    'classwkr', 'hwsei', 'year', 'serial', 'pernum', 'hhwt',
    'cluster', 'strata', 'gq', 'sample', 'cbserial', 'histid',
    'multyear', 'sploc', 'sprule', 'perwt', 'citizenship_status',
    'origin_development_level', 'origin_region', 'bpld',
    'age_at_arrival', 'years_in_us', 'immigrant_x_education',
    'bpl', 'immigrant_x_english', 'yrnatur', 'yrimmig',
    'is_naturalized', 'immigrant_x_stem', 'tenure_x_education'
]


# ============================================================
# MAIN STRATIFIED ANALYSIS FUNCTION
# ============================================================

def run_stratified_analysis(
        data: pd.DataFrame,
        strata_dict: Dict,
        analysis_name: str,
        treatment_col: str = 'is_immigrant',
        outcome_col: str = 'occscore',
        min_immigrants: int = 50,
        sample_natives: Optional[int] = None,
        verbose: bool = True
) -> pd.DataFrame:
    """
    Run DML across all strata and compile results.

    MEMORY-OPTIMIZED VERSION - works with pre-loaded pandas DataFrame.

    Args:
        data: Pre-loaded pandas DataFrame (can be sampled for testing)
        strata_dict: Dictionary of {stratum_name: filter_function}
        analysis_name: Name of this analysis (for printing)
        treatment_col: Treatment variable
        outcome_col: Outcome variable
        min_immigrants: Minimum immigrants required per stratum
        sample_natives: If set, sample this many natives per stratum (saves memory)
        verbose: Print progress

    Returns:
        DataFrame with results for each stratum
    """

    print(f"\n{'=' * 70}")
    print(f"STRATIFIED ANALYSIS: {analysis_name}")
    print(f"{'=' * 70}")

    df = data  # Already loaded pandas DataFrame
    print(f"Working with {len(df):,} observations")

    # Separate immigrants and natives using pandas syntax
    immigrants = df[df[treatment_col] == 1].copy()
    natives = df[df[treatment_col] == 0].copy()

    print(f"Immigrants: {len(immigrants):,}")
    print(f"Natives: {len(natives):,}")

    # Optionally sample natives to save memory
    if sample_natives and len(natives) > sample_natives:
        print(f"  Sampling {sample_natives:,} natives (from {len(natives):,}) to save memory")
        natives = natives.sample(n=sample_natives, random_state=42)

    # Results storage
    results = []

    # Run DML for each stratum
    for stratum_name, filter_fn in strata_dict.items():
        print(f"\n{'-' * 50}")
        print(f"Stratum: {stratum_name}")
        print(f"{'-' * 50}")

        # Filter immigrants to this stratum (using pandas)
        try:
            immigrants_stratum = filter_fn(immigrants)
            n_immigrants = len(immigrants_stratum)
        except Exception as e:
            print(f"  ✗ Filter error: {e}")
            continue

        if n_immigrants < min_immigrants:
            print(f"  ⚠ Skipping: Only {n_immigrants} immigrants (need >= {min_immigrants})")
            continue

        print(f"  Immigrants in stratum: {n_immigrants:,}")
        print(f"  Natives (comparison): {len(natives):,}")

        # Combine immigrants in stratum with natives (pandas concat)
        stratum_df = pd.concat([immigrants_stratum, natives], ignore_index=True)
        print(f"  Total for DML: {len(stratum_df):,}")

        try:
            print(f"  Running DML...")
            dml_result = run_dml_on_subset(
                df=stratum_df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                exclude_cols=STANDARD_EXCLUDE,
                save_dir=f"./dml_temp/{stratum_name.replace(' ', '_').replace('+', 'plus')}",
                verbose=False
            )

            effect = dml_result['effect']
            se = dml_result['se']
            p_value = dml_result['p_value']

            # Significance stars
            if p_value < 0.001:
                stars = "***"
            elif p_value < 0.01:
                stars = "**"
            elif p_value < 0.05:
                stars = "*"
            else:
                stars = ""

            print(f"  ✓ Effect: {effect:.4f}{stars} (SE: {se:.4f}, p={p_value:.4f})")

            results.append({
                'stratum': stratum_name,
                'n_immigrants': n_immigrants,
                'n_natives': len(natives),
                'n_total': len(stratum_df),
                'effect': effect,
                'se': se,
                'p_value': p_value,
                'ci_lower': dml_result['ci_lower'],
                'ci_upper': dml_result['ci_upper'],
                'significant': p_value < 0.05,
                'formatted': f"{effect:.4f}{stars} ({se:.4f})",
                'outcome_r2': dml_result.get('outcome_r2'),
                'treatment_accuracy': dml_result.get('treatment_accuracy'),
            })

        except Exception as e:
            print(f"  ✗ DML error: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Cleanup
        del stratum_df
        gc.collect()

    # Compile results
    results_df = pd.DataFrame(results)

    # Print summary table
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {analysis_name}")
    print(f"{'=' * 70}")
    print(f"\n{'Stratum':<15} {'N':>10} {'Effect':>15} {'p-value':>12}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['stratum']:<15} {row['n_total']:>10,} {row['formatted']:>15} {row['p_value']:>12.4f}")
    print("-" * 55)

    return results_df


# ============================================================
# CONVENIENCE FUNCTIONS FOR EACH RESEARCH QUESTION
# ============================================================

def analyze_question_3(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Question 3: How does time in the US affect occupational mobility?
    """
    return run_stratified_analysis(
        data=data,
        strata_dict=YEARS_IN_US_STRATA,
        analysis_name="YEARS IN US (ASSIMILATION)",
        **kwargs
    )


def analyze_question_4(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Question 4: Are educational returns different for immigrants?
    """
    return run_stratified_analysis(
        data=data,
        strata_dict=EDUCATION_STRATA,
        analysis_name="EDUCATION LEVEL (RETURNS)",
        **kwargs
    )


def analyze_question_5(data: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Question 5: Which migration cohorts achieve better outcomes?
    """
    return run_stratified_analysis(
        data=data,
        strata_dict=ORIGIN_REGION_STRATA,
        analysis_name="ORIGIN REGION",
        **kwargs
    )


def load_data_with_sampling(
        data_path: str,
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        seed: int = 42
) -> pd.DataFrame:
    """
    Memory-efficient data loading with optional sampling.

    Args:
        data_path: Path to parquet file
        sample_size: Exact number of rows to sample (e.g., 100000)
        sample_fraction: Fraction to sample (e.g., 0.1 for 10%)
        seed: Random seed for reproducibility

    Returns:
        Pandas DataFrame (sampled if requested)
    """
    print(f"Loading data from {data_path}...")

    # Use polars for efficient loading, then convert
    df_pl = pl.read_parquet(data_path)
    total_rows = len(df_pl)
    print(f"  Total rows in file: {total_rows:,}")

    # Apply sampling if requested
    if sample_size:
        n = min(sample_size, total_rows)
        df_pl = df_pl.sample(n=n, seed=seed)
        print(f"  Sampled {n:,} rows ({n / total_rows * 100:.1f}%)")
    elif sample_fraction:
        df_pl = df_pl.sample(fraction=sample_fraction, seed=seed)
        print(f"  Sampled {len(df_pl):,} rows ({sample_fraction * 100:.1f}%)")

    # Convert to pandas
    df = df_pl.to_pandas()

    # Cleanup polars
    del df_pl
    gc.collect()

    print(f"  ✓ Loaded {len(df):,} rows, {len(df.columns)} columns")
    return df


def run_all_analyses(
        data_path: str,
        output_dir: str = "./results",
        sample_size: Optional[int] = None,
        sample_fraction: Optional[float] = None,
        sample_natives_per_stratum: Optional[int] = None,
        min_immigrants: int = 50
) -> Dict[str, pd.DataFrame]:
    """
    Run all three stratified analyses.

    MEMORY-OPTIMIZED: Loads data once, optionally samples.

    Args:
        data_path: Path to parquet file
        output_dir: Where to save results
        sample_size: If set, sample this many rows total (for testing)
        sample_fraction: If set, sample this fraction (e.g., 0.1 for 10%)
        sample_natives_per_stratum: If set, use only this many natives per stratum
        min_immigrants: Minimum immigrants needed per stratum

    Returns:
        Dictionary with results DataFrames
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load data ONCE
    df = load_data_with_sampling(
        data_path=data_path,
        sample_size=sample_size,
        sample_fraction=sample_fraction
    )

    results = {}

    # Common kwargs
    common_kwargs = {
        'min_immigrants': min_immigrants,
        'sample_natives': sample_natives_per_stratum
    }

    # Question 3
    try:
        results['years_in_us'] = analyze_question_3(df, **common_kwargs)
        results['years_in_us'].to_csv(f"{output_dir}/q3_years_in_us.csv", index=False)
    except Exception as e:
        print(f"✗ Question 3 failed: {e}")
        results['years_in_us'] = pd.DataFrame()

    # Question 4
    try:
        results['education'] = analyze_question_4(df, **common_kwargs)
        results['education'].to_csv(f"{output_dir}/q4_education.csv", index=False)
    except Exception as e:
        print(f"✗ Question 4 failed: {e}")
        results['education'] = pd.DataFrame()

    try:
        results['origin_development'] = analyze_question_5(df, **common_kwargs)
        results['origin_development'].to_csv(f"{output_dir}/q5_origin.csv", index=False)
    except Exception as e:
        print(f"✗ Question 5 failed: {e}")
        results['education'] = pd.DataFrame()
    # Cleanup
    del df
    gc.collect()

    # Final summary
    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)



    return results


# ============================================================
# VISUALIZATION
# ============================================================

def plot_forest(results_df: pd.DataFrame, title: str, save_path: str = None):
    """Create forest plot of stratified effects."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    y_pos = range(len(results_df))
    effects = results_df['effect'].values
    ci_lower = results_df['ci_lower'].values
    ci_upper = results_df['ci_upper'].values

    # Error bars
    xerr = [effects - ci_lower, ci_upper - effects]

    ax.errorbar(effects, y_pos, xerr=xerr, fmt='o',
                capsize=5, capthick=2, markersize=10,
                color='steelblue', ecolor='steelblue')

    # Zero line
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(results_df['stratum'])
    ax.set_xlabel('Immigrant Effect on OCCSCORE')
    ax.set_title(title)

    # Add effect labels
    for i, (effect, sig) in enumerate(zip(effects, results_df['significant'])):
        marker = "*" if sig else ""
        ax.annotate(f"{effect:.3f}{marker}", (effect, i),
                    textcoords="offset points", xytext=(10, 0),
                    fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    DATA_PATH = "ipums_data_training_final.parquet"
    OUTPUT_DIR = "./stratified_results"

    # ========================================
    # OPTION 1: Quick test with small sample
    # ========================================
    # results = run_all_analyses(
    #     DATA_PATH,
    #     OUTPUT_DIR,
    #     sample_size=50000,  # Only 50k rows for testing
    #     min_immigrants=30   # Lower threshold for small sample
    # )

    # ========================================
    # OPTION 2: 10% sample (memory-friendly)
    # ========================================
    results = run_all_analyses(
        DATA_PATH,
        OUTPUT_DIR,
        sample_fraction=0.1,  # 10% of data
        min_immigrants=50
    )

    # ========================================
    # OPTION 3: Full data with native sampling
    # ========================================
    # results = run_all_analyses(
    #     DATA_PATH,
    #     OUTPUT_DIR,
    #     sample_natives_per_stratum=100000,  # Limit natives per stratum
    #     min_immigrants=100
    # )

    # Generate plots
    for name, result_df in results.items():
        if len(result_df) > 0:
            plot_forest(
                result_df,
                f"Immigrant Penalty: {name.replace('_', ' ').title()}",
                f"{OUTPUT_DIR}/{name}_forest.png"
            )