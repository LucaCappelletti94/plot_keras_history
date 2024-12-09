"""Test submodule to check if suptitle is set correctly."""

import pandas as pd
import matplotlib
from plot_keras_history import plot_history

matplotlib.use("Agg")


def test_suptitle():
    """Test if suptitle is set correctly"""
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0),
        path="plots/test_suptitle.png",
        title="Proviamo a vedere se va",
    )
