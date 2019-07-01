import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import os
import pandas as pd

def test_plot():
    os.makedirs("plots/singletons", exist_ok=True)
    df = pd.read_csv("tests/big_history.csv", index_col=0)
    plot_history(df)
    plt.close()
    df = pd.read_json("tests/small_history.json")
    plot_history(df)
    plt.close()
    df = pd.read_json("tests/history.json")
    plot_history(df, path="plots/singletons", single_graphs=True)
    plt.close()
    plot_history(df, path="plots/normal.png")
    plt.close()
    plot_history(df, path="plots/interpolated.png", interpolate=True)
    plt.close()