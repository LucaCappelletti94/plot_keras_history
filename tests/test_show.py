"""Test submodule to check if show_history works correctly."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from plot_keras_history import show_history


def test_show():
    """Test if show_history works correctly."""
    show_history(pd.read_csv("tests/big_history.csv", index_col=0))
    plt.close()
