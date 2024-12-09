"""Submodule for testing chain_histories function."""
import json
import pandas as pd
from plot_keras_history import chain_histories


def test_chain_histories():
    """Test chain_histories function."""
    history = json.load(open("tests/history.json", "r", encoding="utf-8"))
    double_history = json.load(open("tests/double_history.json", "r", encoding="utf-8"))
    pd.testing.assert_frame_equal(pd.DataFrame(history), chain_histories(history))
    pd.testing.assert_frame_equal(
        pd.DataFrame(double_history), chain_histories(pd.DataFrame(history), history)
    )
