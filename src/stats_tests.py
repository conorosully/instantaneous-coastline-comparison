"""Dataframe-based helpers for the paired significance tests in results.ipynb."""

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def median_summary(model_scores, metrics):
    """Median and sample size per (metric, model) group.

    model_scores : dict[label -> per-image score dataframe]
    metrics      : metric column names to summarise
    """
    rows = []
    for metric in metrics:
        for label, df in model_scores.items():
            vals = df[metric].values
            rows.append({'metric': metric, 'model': label, 'median': np.median(vals), 'n': len(vals)})
    return pd.DataFrame(rows)


def friedman_summary(model_scores, metrics):
    """Friedman test across all groups in model_scores, one row per metric."""
    rows = []
    for metric in metrics:
        groups = [df[metric].values for df in model_scores.values()]
        stat, p = friedmanchisquare(*groups)
        rows.append({'metric': metric, 'n_groups': len(groups), 'statistic': stat, 'p_value': p})
    return pd.DataFrame(rows)


def wilcoxon_vs_baseline(model_scores, metrics, baseline, bonferroni_n=None):
    """Paired Wilcoxon test of every group against `baseline`, Bonferroni-corrected.

    model_scores : dict[label -> per-image score dataframe]
    metrics      : metric column names to test
    baseline     : key in model_scores to compare every other group against
    bonferroni_n : divisor for the significance threshold (defaults to len(model_scores) - 1)
    """
    if bonferroni_n is None:
        bonferroni_n = len(model_scores) - 1
    alpha = 0.05 / bonferroni_n

    rows = []
    baseline_df = model_scores[baseline]
    for metric in metrics:
        baseline_vals = baseline_df[metric].values
        for label, df in model_scores.items():
            if label == baseline:
                continue
            stat, p = wilcoxon(baseline_vals, df[metric].values)
            rows.append({
                'metric':      metric,
                'model':       label,
                'baseline':    baseline,
                'alpha':       alpha,
                'wilcoxon_p':  p,
                'significant': p < alpha,
            })
    return pd.DataFrame(rows)
