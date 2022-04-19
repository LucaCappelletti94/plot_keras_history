import warnings
import compress_json
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def test_warning():
    with warnings.catch_warnings(record=True) as w:
        plot_history(compress_json.local_load(
            "wrong_history.json"
        ))
        assert sum(["normalized" in str(e.message) for e in w]) == 2
