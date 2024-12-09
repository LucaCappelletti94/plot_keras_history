"""Test submodule to check if singletons are plotted."""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from plot_keras_history import plot_history


def test_plot_singletons():
    """Test if singletons are plotted correctly."""
    os.makedirs("plots/singletons", exist_ok=True)
    plot_history(
        pd.read_json("tests/history.json"), path="plots/singletons", single_graphs=True
    )
    plt.close()
