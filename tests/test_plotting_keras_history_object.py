"""Test for the plot_keras_history function."""

from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from plot_keras_history import plot_history

matplotlib.use("Agg")


def test_plotting_keras_history_object():
    """Test if plot_history works correctly."""
    model = Sequential([Dense(1, activation="sigmoid")])
    model.compile(optimizer="nadam", loss="binary_crossentropy")
    X = np.random.uniform(size=(100, 100))
    y = np.random.randint(2, size=(100))
    plot_history(
        model.fit(
            X[:50], y[:50], validation_data=(X[50:], y[50:]), epochs=10, verbose=False
        )
    )
    plt.close()
