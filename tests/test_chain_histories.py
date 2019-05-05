from plot_keras_history import chain_histories
import json

def test_chain_histories():
    history = json.load(open("tests/history.json", "r"))
    double_history = json.load(open("tests/double_history.json", "r"))
    assert history == chain_histories(history, None)
    assert double_history == chain_histories(history, history)