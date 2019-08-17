import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import os
import pandas as pd

def test_plot_small_history():
    plot_history(pd.read_json("tests/small_history.json"), path="plots/small_history.png")
    plt.close()
    assert os.path.exists("plots/small_history.png")