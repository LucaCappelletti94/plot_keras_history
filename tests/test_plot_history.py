from plot_keras_history import plot_history
import json

def test_plot_history():
    history = json.load(open("tests/history.json", "r"))
    plot_history(history)