import warnings
import pytest
import compress_json
from plot_keras_history import plot_history
from plot_keras_history import chain_histories
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_illegal_parameters():
    with pytest.raises(ValueError):
        plot_history(
            compress_json.local_load("wrong_history.json"), monitor_mode="kebab"
        )

    with pytest.raises(ValueError):
        plot_history(
            compress_json.local_load("wrong_history.json"),
            interpolate=True,
            monitor="lr",
        )

    with pytest.raises(ValueError):
        plot_history(compress_json.local_load("wrong_history.json"), max_epochs="kebab")

    with pytest.raises(ValueError):
        chain_histories()
