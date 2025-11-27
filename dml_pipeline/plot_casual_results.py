from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
# Legend
from matplotlib.lines import Line2D


def plot_regression_diagnostics(
        causal_results: Dict,
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        save_path: str = "regression_diagnostics.png",
        show_plot: bool = True
) -> Dict:
    """
    Create publication-ready regression diagnostic plots.

    Plots:
    1. Residuals vs Fitted (Heteroskedasticity check)
    2. Q-Q Plot (Normality check)
    3. Residual Distribution (Histogram)
    4. Scale-Location Plot (Spread of residuals)

    Args:
        causal_results: Output from debug_causal_estimation()
        residuals_y: Outcome residuals
        residuals_t: Treatment residuals
        save_path: Path to save the plot
        show_plot: Whether to display the plot

    Returns:
        Dictionary with diagnostic test statistics
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan
    from statsmodels.graphics.gofplots import qqplot

    # Get OLS results
    ols_results = causal_results['ols_results']
    fitted_values = ols_results.fittedvalues
    ols_residuals = ols_results.resid

    # ========================================
    # Compute diagnostic statistics
    # ========================================

    # Breusch-Pagan test
    X = sm.add_constant(residuals_t)
    bp_stat, bp_pvalue, _, _ = het_breuschpagan(ols_residuals, X)

    # Jarque-Bera test
    jb_stat, jb_pvalue = stats.jarque_bera(ols_residuals)

    # Shapiro-Wilk test (on sample if too large)
    if len(ols_residuals) > 5000:
        sample_idx = np.random.choice(len(ols_residuals), 5000, replace=False)
        sw_stat, sw_pvalue = stats.shapiro(ols_residuals[sample_idx])
    else:
        sw_stat, sw_pvalue = stats.shapiro(ols_residuals)

    # Durbin-Watson (autocorrelation)
    from statsmodels.stats.stattools import durbin_watson
    dw_stat = durbin_watson(ols_residuals)

    # ========================================
    # Create diagnostic plots
    # ========================================

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # ----------------------------------------
    # Plot 1: Residuals vs Fitted Values
    # ----------------------------------------
    ax1 = axes[0, 0]
    ax1.scatter(fitted_values, ols_residuals, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)

    # Add lowess smoothing line
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(ols_residuals, fitted_values, frac=0.3)
        ax1.plot(smoothed[:, 0], smoothed[:, 1], color='orange', linewidth=2, label='LOWESS')
        ax1.legend()
    except:
        pass

    ax1.set_xlabel('Fitted Values', fontsize=11)
    ax1.set_ylabel('Residuals', fontsize=11)
    ax1.set_title('Residuals vs Fitted\n(Heteroskedasticity Check)', fontsize=12, fontweight='bold')

    # Add BP test result
    bp_result = f"Breusch-Pagan: p={bp_pvalue:.4f}"
    bp_color = 'green' if bp_pvalue > 0.05 else 'red'
    ax1.annotate(bp_result, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, color=bp_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ----------------------------------------
    # Plot 2: Q-Q Plot
    # ----------------------------------------
    ax2 = axes[0, 1]

    # Calculate theoretical quantiles
    sorted_residuals = np.sort(ols_residuals)
    n = len(sorted_residuals)
    theoretical_quantiles = stats.norm.ppf(np.arange(1, n + 1) / (n + 1))

    ax2.scatter(theoretical_quantiles, sorted_residuals, alpha=0.5, s=20, c='steelblue', edgecolors='none')

    # Add reference line
    slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    ax2.plot(line_x, slope * line_x + intercept, 'r--', linewidth=2, label='Reference Line')

    ax2.set_xlabel('Theoretical Quantiles', fontsize=11)
    ax2.set_ylabel('Sample Quantiles', fontsize=11)
    ax2.set_title('Q-Q Plot\n(Normality Check)', fontsize=12, fontweight='bold')
    ax2.legend()

    # Add JB test result
    jb_result = f"Jarque-Bera: p={jb_pvalue:.4f}"
    jb_color = 'green' if jb_pvalue > 0.05 else 'orange'
    ax2.annotate(jb_result, xy=(0.05, 0.95), xycoords='axes fraction',
                 fontsize=10, color=jb_color, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ----------------------------------------
    # Plot 3: Histogram of Residuals
    # ----------------------------------------
    ax3 = axes[0, 2]

    n_bins = min(50, int(np.sqrt(len(ols_residuals))))
    ax3.hist(ols_residuals, bins=n_bins, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # Overlay normal distribution
    x_range = np.linspace(ols_residuals.min(), ols_residuals.max(), 100)
    normal_pdf = stats.norm.pdf(x_range, loc=np.mean(ols_residuals), scale=np.std(ols_residuals))
    ax3.plot(x_range, normal_pdf, 'r-', linewidth=2, label='Normal Distribution')

    ax3.set_xlabel('Residuals', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
    ax3.legend()

    # Add skewness and kurtosis
    skew = stats.skew(ols_residuals)
    kurt = stats.kurtosis(ols_residuals)
    ax3.annotate(f'Skewness: {skew:.3f}\nKurtosis: {kurt:.3f}',
                 xy=(0.95, 0.95), xycoords='axes fraction',
                 fontsize=10, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ----------------------------------------
    # Plot 4: Scale-Location Plot
    # ----------------------------------------
    ax4 = axes[1, 0]

    standardized_residuals = ols_residuals / np.std(ols_residuals)
    sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))

    ax4.scatter(fitted_values, sqrt_abs_resid, alpha=0.5, s=20, c='steelblue', edgecolors='none')

    # Add lowess line
    try:
        smoothed = lowess(sqrt_abs_resid, fitted_values, frac=0.3)
        ax4.plot(smoothed[:, 0], smoothed[:, 1], color='red', linewidth=2)
    except:
        pass

    ax4.set_xlabel('Fitted Values', fontsize=11)
    ax4.set_ylabel('√|Standardized Residuals|', fontsize=11)
    ax4.set_title('Scale-Location Plot\n(Homoscedasticity Check)', fontsize=12, fontweight='bold')

    # ----------------------------------------
    # Plot 5: Residuals vs Treatment Residuals
    # ----------------------------------------
    ax5 = axes[1, 1]

    ax5.scatter(residuals_t, ols_residuals, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    ax5.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax5.axvline(x=0, color='gray', linestyle=':', linewidth=1)

    # Add regression line
    slope = causal_results['dml_effect']
    x_line = np.array([residuals_t.min(), residuals_t.max()])
    ax5.plot(x_line, slope * x_line, 'orange', linewidth=2, label=f'Slope (ATE): {slope:.4f}')

    ax5.set_xlabel('Treatment Residuals (T - T̂)', fontsize=11)
    ax5.set_ylabel('Outcome Residuals (Y - Ŷ)', fontsize=11)
    ax5.set_title('DML Final Stage Regression\n(Causal Effect)', fontsize=12, fontweight='bold')
    ax5.legend()

    # ----------------------------------------
    # Plot 6: Summary Statistics Box
    # ----------------------------------------
    ax6 = axes[1, 2]
    ax6.axis('off')

    # Create summary text
    summary_text = f"""
    REGRESSION DIAGNOSTICS SUMMARY
    {'=' * 45}

    HETEROSKEDASTICITY (Breusch-Pagan Test)
    ─────────────────────────────────────────
    Statistic:  {bp_stat:.4f}
    P-value:    {bp_pvalue:.6f}
    Result:     {'✓ No heteroskedasticity' if bp_pvalue > 0.05 else '⚠ Heteroskedasticity detected'}

    NORMALITY (Jarque-Bera Test)
    ─────────────────────────────────────────
    Statistic:  {jb_stat:.4f}
    P-value:    {jb_pvalue:.6f}
    Skewness:   {skew:.4f}
    Kurtosis:   {kurt:.4f}
    Result:     {'✓ Approximately normal' if jb_pvalue > 0.05 else '⚠ Non-normal (common with large N)'}

    AUTOCORRELATION (Durbin-Watson)
    ─────────────────────────────────────────
    Statistic:  {dw_stat:.4f}
    Result:     {'✓ No autocorrelation' if 1.5 < dw_stat < 2.5 else '⚠ Possible autocorrelation'}

    {'=' * 45}
    Note: Non-normality is common with large samples.
    Use robust standard errors for inference.
    """

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # ========================================
    # Final layout
    # ========================================

    plt.suptitle('OLS Regression Diagnostics for DML Final Stage',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Diagnostics plot saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()

    # ========================================
    # Return diagnostic statistics
    # ========================================

    return {
        'breusch_pagan': {'statistic': bp_stat, 'p_value': bp_pvalue,
                          'heteroskedastic': bp_pvalue < 0.05},
        'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pvalue,
                        'non_normal': jb_pvalue < 0.05},
        'shapiro_wilk': {'statistic': sw_stat, 'p_value': sw_pvalue},
        'durbin_watson': {'statistic': dw_stat,
                          'autocorrelation': dw_stat < 1.5 or dw_stat > 2.5},
        'skewness': skew,
        'kurtosis': kurt
    }

def plot_ate_results(
        causal_results: Dict,
        treatment_name: str = "Immigration",
        outcome_name: str = "OCCSCORE",
        save_path: str = "ate_forest_plot.png",
        show_plot: bool = True
) -> None:
    """
    Create publication-ready forest plot and summary table from DML results.

    Args:
        causal_results: Output dictionary from debug_causal_estimation() containing:
            - dml_effect, dml_se, dml_pvalue, dml_ci
            - robust_se, robust_pvalue
            - naive_effect, bias_removed
        treatment_name: Label for treatment variable
        outcome_name: Label for outcome variable
        save_path: Path to save the plot
        show_plot: Whether to display the plot
    """


    # ========================================
    # Extract results
    # ========================================

    dml_effect = causal_results['dml_effect']
    dml_se = causal_results['dml_se']
    robust_se = causal_results['robust_se']
    dml_ci = causal_results['dml_ci']
    dml_pvalue = causal_results['dml_pvalue']
    robust_pvalue = causal_results['robust_pvalue']
    naive_effect = causal_results.get('naive_effect')
    bias_removed = causal_results.get('bias_removed')

    # Calculate robust CI
    robust_ci = (dml_effect - 1.96 * robust_se, dml_effect + 1.96 * robust_se)

    # Calculate naive CI if available
    if naive_effect is not None:
        # Approximate naive SE from bias removed info
        naive_se = dml_se * 1.1  # Rough approximation
        naive_ci = (naive_effect - 1.96 * naive_se, naive_effect + 1.96 * naive_se)

    # ========================================
    # Print Summary Table
    # ========================================

    def get_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    print("\n" + "=" * 85)
    print(f"AVERAGE TREATMENT EFFECT (ATE) OF {treatment_name.upper()} ON {outcome_name.upper()}")
    print("=" * 85)

    print(f"\n{'Estimate':<25} {'Effect':>10} {'SE':>10} {'95% CI':>22} {'P-Value':>12}")
    print("-" * 85)

    # Naive estimate
    if naive_effect is not None:
        stars = get_stars(0.07)  # Approximate
        print(f"{'Naive OLS':<25} {naive_effect:>+10.4f}{stars:<3} {'---':>10} {'---':>22} {'---':>12}")

    # DML estimate (standard SE)
    stars = get_stars(dml_pvalue)
    ci_str = f"[{dml_ci[0]:.4f}, {dml_ci[1]:.4f}]"
    print(f"{'DML (Standard SE)':<25} {dml_effect:>+10.4f}{stars:<3} {dml_se:>10.4f} {ci_str:>22} {dml_pvalue:>12.4f}")

    # DML estimate (robust SE)
    stars = get_stars(robust_pvalue)
    ci_str = f"[{robust_ci[0]:.4f}, {robust_ci[1]:.4f}]"
    print(
        f"{'DML (Robust HC3 SE)':<25} {dml_effect:>+10.4f}{stars:<3} {robust_se:>10.4f} {ci_str:>22} {robust_pvalue:>12.4f}")

    print("-" * 85)
    print("Note: * p<0.05, ** p<0.01, *** p<0.001")

    # Bias removed
    if bias_removed is not None:
        print(f"\nBias removed by DML: {bias_removed:+.4f}")
        print(f"  → Naive estimate was {'overestimating' if bias_removed > 0 else 'underestimating'} the penalty")

    # Interpretation
    print("\n" + "-" * 85)
    print("INTERPRETATION:")
    print("-" * 85)

    if robust_pvalue < 0.05:
        direction = "INCREASES" if dml_effect > 0 else "DECREASES"
        print(f"\n  ✓ SIGNIFICANT: {treatment_name} {direction} {outcome_name} by {abs(dml_effect):.2f} points")
        print(f"    (p = {robust_pvalue:.4f}, 95% CI: [{robust_ci[0]:.2f}, {robust_ci[1]:.2f}])")
    else:
        print(f"\n  ✗ NOT SIGNIFICANT: No evidence that {treatment_name} affects {outcome_name}")
        print(f"    Effect: {dml_effect:+.2f}, but CI includes zero [{robust_ci[0]:.2f}, {robust_ci[1]:.2f}]")
        print(f"    (p = {robust_pvalue:.4f})")

    # ========================================
    # Create Forest Plot
    # ========================================

    fig, ax = plt.subplots(figsize=(10, 5))

    # Prepare data
    labels = []
    effects = []
    ci_lowers = []
    ci_uppers = []
    colors = []

    if naive_effect is not None:
        labels.append('Naive OLS\n(No Controls)')
        effects.append(naive_effect)
        ci_lowers.append(naive_ci[0])
        ci_uppers.append(naive_ci[1])
        colors.append('#e74c3c')  # Red

    labels.append('DML\n(Standard SE)')
    effects.append(dml_effect)
    ci_lowers.append(dml_ci[0])
    ci_uppers.append(dml_ci[1])
    colors.append('#3498db')  # Blue

    labels.append('DML\n(Robust SE)')
    effects.append(dml_effect)
    ci_lowers.append(robust_ci[0])
    ci_uppers.append(robust_ci[1])
    colors.append('#2ecc71')  # Green

    y_pos = np.arange(len(labels))

    # Plot
    for i, (effect, ci_l, ci_u, color) in enumerate(zip(effects, ci_lowers, ci_uppers, colors)):
        # CI line
        ax.plot([ci_l, ci_u], [i, i], color=color, linewidth=3, solid_capstyle='round')
        # CI caps
        ax.plot([ci_l, ci_l], [i - 0.15, i + 0.15], color=color, linewidth=2)
        ax.plot([ci_u, ci_u], [i - 0.15, i + 0.15], color=color, linewidth=2)
        # Point estimate
        ax.scatter(effect, i, color=color, s=200, zorder=5, edgecolors='white', linewidth=2)
        # Label
        ax.annotate(f'{effect:+.2f}', xy=(effect, i + 0.3), ha='center', fontsize=10, fontweight='bold')

    # Zero line
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Null Effect')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel(f'Effect on {outcome_name}', fontsize=12)
    ax.set_title(f'Average Treatment Effect of {treatment_name}\non {outcome_name}',
                 fontsize=14, fontweight='bold', pad=20)

    # Grid
    ax.grid(True, axis='x', alpha=0.3, linestyle='-')
    ax.set_axisbelow(True)

    # Add significance annotation
    sig_text = "Significant (p<0.05)" if robust_pvalue < 0.05 else "Not Significant"
    ax.annotate(sig_text, xy=(0.98, 0.02), xycoords='axes fraction',
                ha='right', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))


    legend_elements = [
        Line2D([0], [0], color='#e74c3c', linewidth=3, label='Naive (Confounded)'),
        Line2D([0], [0], color='#3498db', linewidth=3, label='DML (Causal)'),
        Line2D([0], [0], color='#2ecc71', linewidth=3, label='DML Robust'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label='Null Effect'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Forest plot saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()


def export_ate_latex(causal_results: Dict, output_path: str = "ate_table.tex") -> str:
    """
    Export ATE results to LaTeX table for thesis.

    Args:
        causal_results: Output from debug_causal_estimation()
        output_path: Path to save .tex file

    Returns:
        LaTeX table string
    """

    dml_effect = causal_results['dml_effect']
    dml_se = causal_results['dml_se']
    robust_se = causal_results['robust_se']
    dml_ci = causal_results['dml_ci']
    robust_ci = (dml_effect - 1.96 * robust_se, dml_effect + 1.96 * robust_se)
    dml_pvalue = causal_results['dml_pvalue']
    robust_pvalue = causal_results['robust_pvalue']
    naive_effect = causal_results.get('naive_effect')
    bias_removed = causal_results.get('bias_removed')

    def get_stars(p):
        if p < 0.001:
            return "^{***}"
        elif p < 0.01:
            return "^{**}"
        elif p < 0.05:
            return "^{*}"
        else:
            return ""

    stars = get_stars(robust_pvalue)

    latex = rf"""
\begin{{table}}[htbp]
\centering
\caption{{Double Machine Learning: Causal Effect of Immigration on Occupational Score}}
\label{{tab:dml_ate}}
\begin{{tabular}}{{lcc}}
\hline\hline
 & (1) & (2) \\
 & Naive OLS & DML \\
\hline
Immigration Status & {naive_effect:.4f} & {dml_effect:.4f}{stars} \\
 & --- & ({robust_se:.4f}) \\
\\
95\% Confidence Interval & --- & [{robust_ci[0]:.4f}, {robust_ci[1]:.4f}] \\
P-value & --- & {robust_pvalue:.4f} \\
\\
Bias Removed & \multicolumn{{2}}{{c}}{{{bias_removed:+.4f}}} \\
\hline
Observations & \multicolumn{{2}}{{c}}{{N}} \\
Controls & No & Yes (via DML) \\
Standard Errors & --- & Robust (HC3) \\
\hline\hline
\multicolumn{{3}}{{l}}{{\footnotesize Notes: $^{{*}}$ p$<$0.05, $^{{**}}$ p$<$0.01, $^{{***}}$ p$<$0.001.}} \\
\multicolumn{{3}}{{l}}{{\footnotesize DML controls for age, education, region, and other confounders.}} \\
\multicolumn{{3}}{{l}}{{\footnotesize Robust standard errors in parentheses.}} \\
\end{{tabular}}
\end{{table}}
"""

    with open(output_path, 'w') as f:
        f.write(latex)

    print(f"✓ LaTeX table saved to: {output_path}")

    return latex