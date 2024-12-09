"""Test submodule to check if warnings are raised."""
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import compress_json
import pandas as pd


def test_plot_small_history():
    """Test if small history is plotted correctly."""
    plot_history(
        pd.read_json("tests/small_history.json"), path="plots/small_history.png"
    )
    plt.close()
    assert os.path.exists("plots/small_history.png")
    plot_history("tests/small_history.json")
    plt.close()
    plot_history(["tests/small_history.json", "tests/small_history.json"])
    plt.close()
    plot_history(compress_json.load("tests/small_history.json"))
    plt.close()
