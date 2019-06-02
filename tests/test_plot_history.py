import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import json


def test_plot_history():
    plot_history(json.load(open("tests/history.json", "r")))
    plt.savefig("test_plot_history.png")
    plt.close()