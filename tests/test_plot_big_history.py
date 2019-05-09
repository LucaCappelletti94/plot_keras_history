from plot_keras_history import plot_history
import json

def test_plot_big_history():
    history = json.load(open("tests/big_history.json", "r"))
    plot_history(history)