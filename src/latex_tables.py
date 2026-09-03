"""Formatting helpers shared by the LaTeX table-building cells in results.ipynb."""

import numpy as np


def format_metric(val, bold=False):
    """Format a metric value to 3 decimal places, or '---' if missing/NaN."""
    if val is None:
        return '---'
    try:
        v = float(val)
    except (TypeError, ValueError):
        return '---'
    if np.isnan(v):
        return '---'
    s = f'{v:.3f}'
    return rf'\textbf{{{s}}}' if bold else s


def multirow(span, text):
    """LaTeX \\multirow cell spanning `span` rows, or plain/empty text otherwise."""
    if span > 1:
        return rf'\multirow{{{span}}}{{*}}{{{text}}}'
    return text if span == 1 else ''
