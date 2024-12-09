"""Tests for plot_history function."""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from plot_keras_history import plot_history


def test_plot():
    """Test if plot_history works correctly."""
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0)[:16],
        path="plots/interpolated.png",
        interpolate=True,
    )
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0)[:2],
        path="plots/interpolated.png",
        interpolate=True,
    )
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0),
        path="plots/interpolated.png",
        interpolate=True,
    )
    plt.close()
    assert os.path.exists("plots/interpolated.png")
