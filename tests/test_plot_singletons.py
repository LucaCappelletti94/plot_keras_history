import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import os
import pandas as pd


def test_plot_singletons():
    os.makedirs("plots/singletons", exist_ok=True)
    plot_history(
        pd.read_json("tests/history.json"), path="plots/singletons", single_graphs=True
    )
    plt.close()
