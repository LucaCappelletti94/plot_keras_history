import pandas as pd
import os
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_suptitle():
    plot_history(
        pd.read_csv("tests/big_history.csv", index_col=0),
        path="plots/test_suptitle.png",
        title="Proviamo a vedere se va",
    )
