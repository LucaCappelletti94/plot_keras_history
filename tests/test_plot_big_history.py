import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_keras_history import plot_history
import json

def test_plot_big_history():
    plot_history(json.load(open("tests/big_history.json", "r")))
    plt.savefig("test_plot_big_history.png")
    plt.close()