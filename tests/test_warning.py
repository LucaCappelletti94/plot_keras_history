"""Test submodule to check if warnings are raised."""

import warnings
import compress_json
import matplotlib
from plot_keras_history import plot_history

matplotlib.use("Agg")


def test_warning():
    """Test if warnings are raised."""
    with warnings.catch_warnings(record=True) as w:
        plot_history(compress_json.local_load("wrong_history.json"))
        assert sum(["normalized" in str(e.message) for e in w]) == 3
