"""Test the plot_history function with a big history."""

import pandas as pd
import os
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_plot_big_history():
    """Test if plot_history works correctly when plotting a big history."""
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0),
        path="plots/big_history.png",
        log_scale_metrics=True,
    )
    plt.close()
    assert os.path.exists("plots/big_history.png")
