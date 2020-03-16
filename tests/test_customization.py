import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import os
import pandas as pd
import pytest

def callami(axis):
    axis.set_title("TESTONI")

def test_customization():
    plot_history(pd.read_csv("tests/big_history.csv", index_col=0), path="plots/normal.png", customization_callback=callami)
    plt.close()
    plot_history("tests/big_history.csv")
    plt.close()
    plot_history(["tests/big_history.csv", "tests/big_history.csv"])
    plt.close()

    with pytest.raises(ValueError):
        plot_history(78)