"""Test submodule for plotting multiple histories."""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from plot_keras_history import plot_history

matplotlib.use("Agg")


def test_plot_multi_history():
    """Test if plot_history works correctly when plotting multiple histories."""
    plot_history(
        [
            pd.read_csv("tests/history1.csv"),
            pd.read_csv("tests/history2.csv"),
            pd.read_csv("tests/history4.csv"),
            pd.read_csv("tests/history3.csv"),
        ],
        path="plots/multiple.png",
        interpolate=True,
        max_epochs="min",
        show_standard_deviation=True,
    )
    plt.close()
    plot_history(
        [
            pd.read_csv("tests/history1.csv"),
            pd.read_csv("tests/history2.csv"),
            pd.read_csv("tests/history4.csv"),
            pd.read_csv("tests/history3.csv"),
        ],
        path="plots/multiple.png",
        interpolate=True,
        max_epochs="max",
    )
    plt.close()
    assert os.path.exists("plots/multiple.png")
