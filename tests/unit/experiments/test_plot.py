"""Minimal smoke tests for plotting utilities."""

import pandas as pd

from motools.experiments.analysis.plot import (
    plot_sweep_heatmap,
    plot_sweep_metric,
    plot_sweep_metrics,
)


def test_plot_sweep_metric():
    """Test basic metric plotting."""
    data = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    fig = plot_sweep_metric(data, x="x", y="y")
    assert fig is not None
    assert len(fig.data) > 0


def test_plot_sweep_metrics():
    """Test plotting multiple metrics."""
    data = pd.DataFrame({"x": [1, 2, 3], "y1": [10, 20, 30], "y2": [15, 25, 35]})
    fig = plot_sweep_metrics(data, x="x", y=["y1", "y2"])
    assert fig is not None


def test_plot_sweep_heatmap():
    """Test heatmap plotting."""
    data = pd.DataFrame({"x": [1, 1, 2, 2], "y": ["a", "b", "a", "b"], "val": [10, 20, 30, 40]})
    fig = plot_sweep_heatmap(data, x="x", y="y", values="val")
    assert fig is not None
    assert hasattr(fig.data[0], "z")
