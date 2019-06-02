import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import json


def test_plot_history():
    plot_history(json.load(open("tests/small_history.json", "r")), interpolate=True)
    plt.savefig("test_plot_small_history.png")
    plt.close()