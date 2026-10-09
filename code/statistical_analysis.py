#!/usr/bin/env python3
"""
Statistical Analysis for Evolutionary Algorithms Paper
Tests significance of differences between methods
"""

import csv
import json
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import f_oneway, ttest_ind, shapiro, levene, mannwhitneyu


def load_data():
    """Load experiment results from CSV"""
    results = []
    with open('experiment_results_detailed.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'method': row['method'],
                'shape': row['shape'],
                'run': int(row['run']),
                'fitness': float(row['fitness']),
                'convergence_gen': float(row['convergence_gen']),
            })
    return results


def filter_by_method_shape(results, method, shape):
    """Filter results for specific method and shape"""
    return [r for r in results if r['method'] == method and r['shape'] == shape]


def normality_test(data, name=""):
    """Shapiro-Wilk test for normality"""
    if len(data) < 3:
        return None, None

    stat, p_value = shapiro(data)
    normal = p_value > 0.05

    return normal, p_value


def levene_test(groups):
    """Levene's test for equal variances"""
    stat, p_value = levene(*groups)
    equal_var = p_value > 0.05

    return equal_var, p_value


def anova_test(results, metric='fitness'):
    """
    One-way ANOVA comparing all methods across all shapes
    H0: All methods have equal mean fitness/convergence
    """
    print("\n" + "="*80)
    print(f"ONE-WAY ANOVA: {metric.upper()}")
    print("="*80)
    print("H0: All methods have equal mean", metric)
    print("H1: At least one method differs significantly\n")

    methods = list(set(r['method'] for r in results))

    # Group data by method
    groups = {}
    for method in methods:
        data = [r[metric] for r in results if r['method'] == method and not np.isnan(r[metric])]
        groups[method] = data

    # Prepare arrays for ANOVA
    arrays = [np.array(groups[m]) for m in methods]

    # ANOVA
    f_stat, p_value = f_oneway(*arrays)

    print(f"Methods: {methods}")
    print(f"\nF-statistic: {f_stat:.4f}")
    print(f"P-value: {p_value:.6f}")

    if p_value < 0.05:
        print("✓ SIGNIFICANT DIFFERENCE (p < 0.05)")
        print("  Reject H0: Methods significantly different!")
    else:
        print("✗ NO SIGNIFICANT DIFFERENCE (p >= 0.05)")
        print("  Fail to reject H0: Methods similar")

    # Print means and std
    print(f"\nDescriptive Statistics:")
    print(f"{'Method':<20} {'Mean':<15} {'Std':<15} {'N':<5}")
    print("-"*55)
    for method in methods:
        data = np.array(groups[method])
        print(f"{method:<20} {np.mean(data):<15.4f} {np.std(data):<15.4f} {len(data):<5}")

    return f_stat, p_value, groups


def pairwise_ttests(groups, metric='fitness', bonferroni=True):
    """
    Pairwise t-tests between methods
    With Bonferroni correction for multiple comparisons
    """
    print("\n" + "="*80)
    print(f"PAIRWISE T-TESTS: {metric.upper()}")
    print("="*80)

    methods = list(groups.keys())
    n_comparisons = len(methods) * (len(methods) - 1) // 2

    if bonferroni:
        alpha = 0.05 / n_comparisons
        print(f"Bonferroni correction: α = {alpha:.4f} (per comparison)")
    else:
        alpha = 0.05
        print(f"No correction: α = {alpha}")

    print(f"Number of pairwise comparisons: {n_comparisons}\n")

    print(f"{'Method 1':<20} {'Method 2':<20} {'t-stat':<12} {'p-value':<12} {'Significant':<12}")
    print("-"*76)

    results = []
    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            data1 = np.array(groups[method1])
            data2 = np.array(groups[method2])

            # Welch's t-test (doesn't assume equal variance)
            t_stat, p_value = ttest_ind(data1, data2, equal_var=False)

            significant = "YES" if p_value < alpha else "NO"
            print(f"{method1:<20} {method2:<20} {t_stat:<12.4f} {p_value:<12.6f} {significant:<12}")

            results.append({
                'method1': method1,
                'method2': method2,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < alpha,
            })

    return results


def effect_size_cohens_d(group1, group2):
    """
    Cohen's d effect size
    0.2 = small, 0.5 = medium, 0.8 = large
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    mean_diff = np.mean(group1) - np.mean(group2)

    return mean_diff / pooled_std if pooled_std != 0 else 0


def effect_sizes(groups, methods=None):
    """Calculate effect sizes between all method pairs"""
    if methods is None:
        methods = list(groups.keys())

    print("\n" + "="*80)
    print("EFFECT SIZES (Cohen's d)")
    print("="*80)
    print("Interpretation: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large\n")

    print(f"{'Method 1':<20} {'Method 2':<20} {'Cohen\'s d':<15} {'Effect Size':<15}")
    print("-"*70)

    for i, method1 in enumerate(methods):
        for method2 in methods[i+1:]:
            d = effect_size_cohens_d(
                np.array(groups[method1]),
                np.array(groups[method2])
            )

            if abs(d) < 0.2:
                effect = "negligible"
            elif abs(d) < 0.5:
                effect = "small"
            elif abs(d) < 0.8:
                effect = "medium"
            else:
                effect = "large"

            print(f"{method1:<20} {method2:<20} {d:<15.4f} {effect:<15}")


def analyze_by_shape(results, metric='fitness'):
    """Separate analysis for each shape"""
    print("\n" + "="*80)
    print(f"ANALYSIS BY SHAPE: {metric.upper()}")
    print("="*80)

    shapes = list(set(r['shape'] for r in results))

    for shape in shapes:
        print(f"\n{'-'*80}")
        print(f"SHAPE: {shape}")
        print(f"{'-'*80}")

        shape_results = filter_by_method_shape(results, None, shape)
        # Filter properly
        shape_results = [r for r in results if r['shape'] == shape]

        # Group by method
        methods = list(set(r['method'] for r in shape_results))
        groups = {}
        for method in methods:
            data = [r[metric] for r in shape_results if r['method'] == method]
            groups[method] = data

        # Stats
        for method in methods:
            data = np.array(groups[method])
            print(f"{method:<20}: mean={np.mean(data):.4f}, std={np.std(data):.4f}, n={len(data)}")

        # ANOVA
        arrays = [np.array(groups[m]) for m in methods]
        f_stat, p_value = f_oneway(*arrays)
        print(f"\nANOVA F={f_stat:.4f}, p={p_value:.6f}", end="")
        print(" → SIGNIFICANT ✓" if p_value < 0.05 else " → Not significant ✗")


def main():
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS: EVOLUTIONARY ALGORITHMS PAPER")
    print("="*80)

    # Load data
    results = load_data()
    print(f"\nLoaded {len(results)} experiment results")

    # Overall ANOVA for fitness
    f_stat_fit, p_fit, groups_fit = anova_test(results, metric='fitness')

    # Overall ANOVA for convergence
    f_stat_conv, p_conv, groups_conv = anova_test(results, metric='convergence_gen')

    # Pairwise t-tests for fitness
    pairwise_fit = pairwise_ttests(groups_fit, metric='fitness', bonferroni=True)

    # Pairwise t-tests for convergence
    pairwise_conv = pairwise_ttests(groups_conv, metric='convergence_gen', bonferroni=True)

    # Effect sizes for fitness
    effect_sizes(groups_fit)

    # Effect sizes for convergence
    effect_sizes(groups_conv)

    # Analysis by shape
    analyze_by_shape(results, metric='fitness')
    analyze_by_shape(results, metric='convergence_gen')

    # Save results
    with open('statistical_results.json', 'w') as f:
        json.dump({
            'anova_fitness': {'f_stat': float(f_stat_fit), 'p_value': float(p_fit)},
            'anova_convergence': {'f_stat': float(f_stat_conv), 'p_value': float(p_conv)},
            'pairwise_fitness': pairwise_fit,
            'pairwise_convergence': pairwise_conv,
        }, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS COMPLETE")
    print("="*80)
    print("Results saved to: statistical_results.json")


if __name__ == "__main__":
    main()
