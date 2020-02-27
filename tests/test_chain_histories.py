from plot_keras_history import chain_histories
import json
import pandas as pd

def test_chain_histories():
    history = json.load(open("tests/history.json", "r"))
    double_history = json.load(open("tests/double_history.json", "r"))
    pd.testing.assert_frame_equal(
        pd.DataFrame(history),
        chain_histories(history, None)
    )
    pd.testing.assert_frame_equal(
        pd.DataFrame(double_history),
        chain_histories(pd.DataFrame(history), history)
    )