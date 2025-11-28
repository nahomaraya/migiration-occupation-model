
import numpy as np
import statsmodels.api as sm

from typing import Dict, List, Optional, Tuple

def estimate_causal_effect(
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        weights: Optional[np.ndarray] = None,
        verbose: bool = True
) -> Dict:
    """
    Step 3: Estimate causal effect using statsmodels OLS.

    Regresses outcome residuals on treatment residuals:
        Unexplained_Success ~ Pure_Immigration_Status

    Produces publication-ready regression table for thesis.

    Args:
        residuals_y: Outcome residuals from Step 1 (Y - Y_hat)
        residuals_t: Treatment residuals from Step 2 (T - T_hat)
        weights: Sample weights (optional)
        verbose: Print regression summary

    Returns:
        Dictionary with causal effect, standard error, p-value,
        confidence intervals, and full OLS summary object
    """
    if verbose:
        print("\n" + "=" * 80)
        print("STEP 3: CAUSAL ESTIMATION (OLS)")
        print("=" * 80)
        print("\n  Model: Unexplained_OCCSCORE ~ Pure_Immigration_Status")
        print("  Method: Ordinary Least Squares (statsmodels)")

    # Prepare data for statsmodels
    # Add constant for intercept
    X = sm.add_constant(residuals_t)
    y = residuals_y

    # Fit OLS model (weighted if weights provided)
    if weights is not None:
        model = sm.WLS(y, X, weights=weights)
    else:
        model = sm.OLS(y, X)

    results = model.fit()

    # Extract key statistics
    causal_effect = results.params[1]  # Coefficient on treatment residual
    std_error = results.bse[1]
    t_stat = results.tvalues[1]
    p_value = results.pvalues[1]
    ci_lower, ci_upper = results.conf_int()[1]
    r_squared = results.rsquared

    # Print publication-ready summary
    if verbose:
        print("\n" + "=" * 80)
        print("                    OLS REGRESSION RESULTS")
        print("                    (Copy-paste for thesis)")
        print("=" * 80)
        print(results.summary())

        print("\n" + "=" * 80)
        print("KEY FINDING: CAUSAL EFFECT OF IMMIGRATION")
        print("=" * 80)
        print(f"\n  Coefficient (β): {causal_effect:.4f}")
        print(f"  Standard Error:  {std_error:.4f}")
        print(f"  t-statistic:     {t_stat:.4f}")
        print(f"  P-value:         {p_value:.6f}")
        print(f"  95% CI:          [{ci_lower:.4f}, {ci_upper:.4f}]")
        print(f"  R-squared:       {r_squared:.4f}")

        # Significance stars (for thesis tables)
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < 0.05:
            stars = "*"
        else:
            stars = ""

        print(f"\n  Formatted for table: {causal_effect:.4f}{stars}")
        print(f"                       ({std_error:.4f})")

        # Interpretation
        print("\n" + "-" * 60)
        print("INTERPRETATION:")
        print("-" * 60)

        if p_value < 0.05:
            if causal_effect < 0:
                print(f"\n  ✓ Statistically significant NEGATIVE effect (p < 0.05)")
                print(f"\n  Being an immigrant DECREASES occupational score by")
                print(f"  {abs(causal_effect):.2f} points, after controlling for")
                print(f"  education, age, region, and other confounders.")
                print(f"\n  This suggests an 'immigrant penalty' in the labor market.")
            else:
                print(f"\n  ✓ Statistically significant POSITIVE effect (p < 0.05)")
                print(f"\n  Being an immigrant INCREASES occupational score by")
                print(f"  {causal_effect:.2f} points, after controlling for")
                print(f"  education, age, region, and other confounders.")
                print(f"\n  This suggests an 'immigrant premium' in the labor market.")
        else:
            print(f"\n  ✗ Effect is NOT statistically significant (p = {p_value:.4f})")
            print(f"\n  We cannot conclude that immigration has a causal effect")
            print(f"  on occupational scores after controlling for confounders.")

    return {
        'causal_effect': causal_effect,
        'standard_error': std_error,
        't_statistic': t_stat,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'r_squared': r_squared,
        'significant_05': p_value < 0.05,
        'significant_01': p_value < 0.01,
        'ols_results': results,  # Full statsmodels results object
        'summary_text': results.summary().as_text()  # For saving to file
    }


def export_results_to_latex(results: Dict, output_path: str = 'dml_results.tex') -> str:
    """
    Export DML results to LaTeX table format for thesis.

    Args:
        results: Output from estimate_causal_effect()
        output_path: Path to save .tex file

    Returns:
        LaTeX table string
    """
    ols = results['ols_results']

    # Generate LaTeX table
    latex_table = ols.summary().as_latex()

    # Save to file
    with open(output_path, 'w') as f:
        f.write(latex_table)

    print(f"✓ LaTeX table saved to: {output_path}")

    # Also create a simple custom table
    effect = results['causal_effect']
    se = results['standard_error']
    p = results['p_value']

    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    simple_latex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{Double Machine Learning: Causal Effect of Immigration on Occupational Score}}
\\label{{tab:dml_results}}
\\begin{{tabular}}{{lc}}
\\hline\\hline
 & OCCSCORE \\\\
\\hline
Immigration Status & {effect:.4f}{stars} \\\\
 & ({se:.4f}) \\\\
\\hline
Observations & {ols.nobs:.0f} \\\\
R-squared & {results['r_squared']:.4f} \\\\
\\hline\\hline
\\multicolumn{{2}}{{l}}{{\\footnotesize Standard errors in parentheses}} \\\\
\\multicolumn{{2}}{{l}}{{\\footnotesize * p < 0.05, ** p < 0.01, *** p < 0.001}} \\\\
\\end{{tabular}}
\\end{{table}}
"""

    # Save simple table
    simple_path = output_path.replace('.tex', '_simple.tex')
    with open(simple_path, 'w') as f:
        f.write(simple_latex)

    print(f"✓ Simple LaTeX table saved to: {simple_path}")

    return simple_latex
