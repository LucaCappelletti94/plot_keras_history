import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import os
import pandas as pd

def test_plot_big_history():
    plot_history(pd.read_csv("tests/big_history.csv", index_col=0))
    plt.close()
    