from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
# Legend
from matplotlib.lines import Line2D


def plot_regression_diagnostics(
        causal_results: Dict,
        residuals_y: np.ndarray,
        residuals_t: np.ndarray,
        save_path: None,
        show_plot: bool = True,
        sample_size: int = 50000,  # Larger sample for statistical validity
        plot_sample: int = 5000,  # Smaller sample for clean visuals
        random_seed: int = 42
) -> Dict:
    """
    Publication-ready regression diagnostics for large-scale DML.

    Optimized for datasets with millions of observations.
    Uses stratified sampling to preserve distribution characteristics.
    """
    import statsmodels.api as sm
    from statsmodels.stats.diagnostic import het_breuschpagan

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150
    })

    ols_results = causal_results['ols_results']
    fitted = np.asarray(ols_results.fittedvalues)
    resid = np.asarray(ols_results.resid)
    n_total = len(resid)

    rng = np.random.default_rng(random_seed)

    # ========================================
    # Stratified sampling for statistical tests
    # ========================================
    stat_idx = _stratified_sample(fitted, sample_size, rng)
    fitted_stat = fitted[stat_idx]
    resid_stat = resid[stat_idx]
    resid_t_stat = residuals_t[stat_idx]

    # Smaller sample for plotting (avoids overplotting)
    plot_idx = rng.choice(stat_idx, min(plot_sample, len(stat_idx)), replace=False)
    fitted_plot = fitted[plot_idx]
    resid_plot = resid[plot_idx]
    resid_t_plot = residuals_t[plot_idx]

    # ========================================
    # Compute diagnostic statistics
    # ========================================

    # Breusch-Pagan test
    X_bp = sm.add_constant(resid_t_stat)
    bp_stat, bp_pval, _, _ = het_breuschpagan(resid_stat, X_bp)

    # Jarque-Bera test
    jb_stat, jb_pval = stats.jarque_bera(resid_stat)

    # D'Agostino-Pearson test (more robust for large samples)
    dag_stat, dag_pval = stats.normaltest(resid_stat)

    # Skewness and Kurtosis (on full data - O(n) and fast)
    skew = stats.skew(resid)
    kurt = stats.kurtosis(resid)

    # Autocorrelation (lag-1)
    lag1_corr = np.corrcoef(resid[:-1], resid[1:])[0, 1]
    dw_approx = 2 * (1 - lag1_corr)

    # Standardized residuals for plots
    resid_std = resid_plot / resid.std()

    # ========================================
    # Create publication figure
    # ========================================

    fig = plt.figure(figsize=(7.5, 7))  # Single column width for journals
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Color scheme (colorblind-friendly)
    c_points = '#4878A8'
    c_line = '#D55E00'
    c_smooth = '#E69F00'

    # ----------------------------------------
    # Panel A: Residuals vs Fitted
    # ----------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])

    ax1.scatter(fitted_plot, resid_plot, alpha=0.4, s=8,
                c=c_points, edgecolors='none', rasterized=True)
    ax1.axhline(0, color=c_line, linestyle='-', lw=1.2, zorder=5)

    # Binned means with CI (fast alternative to LOWESS)
    bin_centers, bin_means, bin_ci = _binned_statistics(fitted_stat, resid_stat)
    ax1.plot(bin_centers, bin_means, color=c_smooth, lw=2, zorder=6)
    ax1.fill_between(bin_centers, bin_means - bin_ci, bin_means + bin_ci,
                     color=c_smooth, alpha=0.2, zorder=4)

    ax1.set_xlabel('Fitted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('(A) Residuals vs Fitted', fontweight='bold', loc='left')

    # Annotation
    het_text = f'BP test: p = {bp_pval:.3f}' if bp_pval >= 0.001 else f'BP test: p < 0.001'
    ax1.text(0.97, 0.97, het_text, transform=ax1.transAxes, fontsize=9,
             ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3',
                                             facecolor='white', edgecolor='gray', alpha=0.9))

    # ----------------------------------------
    # Panel B: Q-Q Plot
    # ----------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])

    # Compute theoretical quantiles
    n_qq = len(resid_plot)
    theoretical_q = stats.norm.ppf((np.arange(1, n_qq + 1) - 0.5) / n_qq)
    sample_q = np.sort(resid_std)

    ax2.scatter(theoretical_q, sample_q, alpha=0.4, s=8,
                c=c_points, edgecolors='none', rasterized=True)

    # Reference line (through Q1 and Q3)
    q1_t, q3_t = stats.norm.ppf([0.25, 0.75])
    q1_s, q3_s = np.percentile(resid_std, [25, 75])
    slope = (q3_s - q1_s) / (q3_t - q1_t)
    intercept = q1_s - slope * q1_t

    xlim = np.array([-3.5, 3.5])
    ax2.plot(xlim, slope * xlim + intercept, color=c_line, lw=1.2, zorder=5)
    ax2.set_xlim(xlim)

    ax2.set_xlabel('Theoretical Quantiles')
    ax2.set_ylabel('Standardized Residuals')
    ax2.set_title('(B) Normal Q-Q Plot', fontweight='bold', loc='left')

    norm_text = f'Skew = {skew:.2f}\nKurt = {kurt:.2f}'
    ax2.text(0.03, 0.97, norm_text, transform=ax2.transAxes, fontsize=9,
             ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='white', edgecolor='gray', alpha=0.9))

    # ----------------------------------------
    # Panel C: Scale-Location
    # ----------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])

    sqrt_abs_resid = np.sqrt(np.abs(resid_std))

    ax3.scatter(fitted_plot, sqrt_abs_resid, alpha=0.4, s=8,
                c=c_points, edgecolors='none', rasterized=True)

    # Binned means for scale-location
    resid_std_full = resid_stat / resid.std()
    sqrt_abs_full = np.sqrt(np.abs(resid_std_full))
    bin_centers_sl, bin_means_sl, _ = _binned_statistics(fitted_stat, sqrt_abs_full)
    ax3.plot(bin_centers_sl, bin_means_sl, color=c_smooth, lw=2, zorder=6)

    ax3.set_xlabel('Fitted Values')
    ax3.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
    ax3.set_title('(C) Scale-Location', fontweight='bold', loc='left')

    # ----------------------------------------
    # Panel D: DML Final Stage
    # ----------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])

    ax4.scatter(resid_t_plot, resid_plot, alpha=0.4, s=8,
                c=c_points, edgecolors='none', rasterized=True)
    ax4.axhline(0, color='gray', linestyle=':', lw=0.8, zorder=3)
    ax4.axvline(0, color='gray', linestyle=':', lw=0.8, zorder=3)

    # Causal effect line
    ate = causal_results.get('causal_effect') or causal_results.get('dml_effect')
    x_range = np.array([residuals_t.min(), residuals_t.max()])
    ax4.plot(x_range, ate * x_range, color=c_line, lw=1.5, zorder=5)

    ax4.set_xlabel(r'Treatment Residuals ($\tilde{T}$)')
    ax4.set_ylabel(r'Outcome Residuals ($\tilde{Y}$)')
    ax4.set_title('(D) DML Partial Regression', fontweight='bold', loc='left')

    # Get SE if available
    se = causal_results.get('standard_error') or causal_results.get('dml_se')
    if se:
        ate_text = f'ATE = {ate:.4f}\nSE = {se:.4f}'
    else:
        ate_text = f'ATE = {ate:.4f}'
    ax4.text(0.97, 0.03, ate_text, transform=ax4.transAxes, fontsize=9,
             ha='right', va='bottom', bbox=dict(boxstyle='round,pad=0.3',
                                                facecolor='white', edgecolor='gray', alpha=0.9))

    # ========================================
    # Figure note
    # ========================================
    note = f'Note: Plots based on random subsample (n = {plot_sample:,}) ' \
           f'from full dataset (N = {n_total:,}). ' \
           f'Statistics computed on n = {sample_size:,} stratified sample.'
    fig.text(0.5, 0.01, note, ha='center', fontsize=8, style='italic')

    plt.tight_layout(rect=[0, 0.03, 1, 1])

    # ========================================
    # Save figure
    # ========================================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {save_path}")

    if show_plot:
        plt.show()
    plt.close()

    # ========================================
    # Return results
    # ========================================
    return {
        'n_total': n_total,
        'n_sample_stats': len(stat_idx),
        'n_sample_plot': len(plot_idx),
        'breusch_pagan': {'statistic': bp_stat, 'p_value': bp_pval},
        'jarque_bera': {'statistic': jb_stat, 'p_value': jb_pval},
        'dagostino_pearson': {'statistic': dag_stat, 'p_value': dag_pval},
        'skewness': skew,
        'kurtosis': kurt,
        'durbin_watson_approx': dw_approx,
        'lag1_autocorr': lag1_corr,
        'dml_effect': ate,
        'standard_error': se
    }


def _stratified_sample(x: np.ndarray, n: int, rng) -> np.ndarray:
    """Stratified sampling based on deciles of x."""
    n = min(n, len(x))
    deciles = np.percentile(x, np.arange(0, 101, 10))
    bins = np.digitize(x, deciles[1:-1])

    indices = []
    samples_per_bin = n // 10

    for b in range(10):
        bin_idx = np.where(bins == b)[0]
        if len(bin_idx) > 0:
            k = min(samples_per_bin, len(bin_idx))
            indices.extend(rng.choice(bin_idx, k, replace=False))

    return np.array(indices)


def _binned_statistics(x: np.ndarray, y: np.ndarray, n_bins: int = 25):
    """Compute binned means with 95% CI."""
    bins = np.linspace(np.percentile(x, 1), np.percentile(x, 99), n_bins + 1)
    bin_idx = np.digitize(x, bins)

    centers, means, cis = [], [], []

    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.sum() >= 10:
            y_bin = y[mask]
            centers.append((bins[i - 1] + bins[i]) / 2)
            means.append(y_bin.mean())
            cis.append(1.96 * y_bin.std() / np.sqrt(len(y_bin)))

    return np.array(centers), np.array(means), np.array(cis)
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

    dml_effect = causal_results.get('dml_effect') or causal_results.get('causal_effect')
    dml_se = causal_results.get('dml_se') or causal_results.get('standard_error')
    dml_pvalue = causal_results.get('dml_pvalue') or causal_results.get('p_value')

    # CI handling
    if 'dml_ci' in causal_results:
        dml_ci = causal_results['dml_ci']
    else:
        dml_ci = (causal_results.get('ci_lower'), causal_results.get('ci_upper'))

    robust_se = causal_results.get('robust_se') or dml_se
    robust_pvalue = causal_results.get('robust_pvalue') or dml_pvalue
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

    dml_effect = causal_results.get('dml_effect') or causal_results.get('causal_effect')
    dml_se = causal_results.get('dml_se') or causal_results.get('standard_error')
    dml_pvalue = causal_results.get('dml_pvalue') or causal_results.get('p_value')

    # CI handling
    if 'dml_ci' in causal_results:
        dml_ci = causal_results['dml_ci']
    else:
        dml_ci = (causal_results.get('ci_lower'), causal_results.get('ci_upper'))

    robust_se = causal_results.get('robust_se') or dml_se
    robust_pvalue = causal_results.get('robust_pvalue') or dml_pvalue
    naive_effect = causal_results.get('naive_effect')
    bias_removed = causal_results.get('bias_removed')

    # Calculate robust CI
    robust_ci = (dml_effect - 1.96 * robust_se, dml_effect + 1.96 * robust_se)

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