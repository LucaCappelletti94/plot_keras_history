"""Test submodule to check if exceptions are raised."""

import pytest
import compress_json
import matplotlib
from plot_keras_history import plot_history
from plot_keras_history import chain_histories

matplotlib.use("Agg")


def test_illegal_parameters():
    """Test if exceptions are raised."""
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
