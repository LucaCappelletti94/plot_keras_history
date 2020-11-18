import pandas as pd
import os
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def test_plot_multi_history():
    plot_history([
        pd.read_csv("tests/history1.csv"),
        pd.read_csv("tests/history2.csv"),
        pd.read_csv("tests/history4.csv"),
        pd.read_csv("tests/history3.csv")
    ], path="plots/multiple.png", interpolate=True, max_epochs="min")
    plt.close()
    plot_history([
        pd.read_csv("tests/history1.csv"),
        pd.read_csv("tests/history2.csv"),
        pd.read_csv("tests/history4.csv"),
        pd.read_csv("tests/history3.csv")
    ], path="plots/multiple.png", interpolate=True, max_epochs="max")
    plt.close()
    assert os.path.exists("plots/multiple.png")
