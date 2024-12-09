"""Tests for the customization_callback parameter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from plot_keras_history import plot_history


def callami(axis):
    """Test callback."""
    axis.set_title("TESTONI")


def test_customization():
    """Test if customization_callback works correctly."""
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0),
        path="plots/normal.png",
        customization_callback=callami,
    )
    plt.close()
    plot_history("tests/big_history.csv")
    plt.close()
    plot_history(["tests/big_history.csv", "tests/big_history.csv"])
    plt.close()

    with pytest.raises(TypeError):
        plot_history(78)
