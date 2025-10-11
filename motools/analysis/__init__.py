"""Analysis utilities for statistics and plotting."""

from motools.analysis.plotting import create_scatterplot, make_ci_plot
from motools.analysis.stats import (
    CI,
    compute_bernoulli_ci,
    compute_ci,
    compute_ci_df,
    compute_probability_ci,
)

__all__ = [
    "CI",
    "compute_ci",
    "compute_bernoulli_ci",
    "compute_probability_ci",
    "compute_ci_df",
    "make_ci_plot",
    "create_scatterplot",
]
