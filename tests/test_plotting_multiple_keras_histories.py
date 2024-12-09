"""Test to check if multiple histories plots look ok."""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from plot_keras_history import plot_history
from extra_keras_metrics import get_minimal_multiclass_metrics
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_plotting_keras_history_object():
    histories = []
    for _ in range(5):
        model = Sequential([Dense(1, activation="sigmoid")])
        model.compile(
            optimizer="nadam",
            loss="binary_crossentropy",
            metrics=get_minimal_multiclass_metrics(),
        )
        size = 1000
        X = np.random.uniform(low=-1, high=+1, size=(size, 100))
        y = np.mean(X, axis=1) > 0
        histories.append(
            model.fit(
                X[: size // 2],
                y[: size // 2],
                batch_size=size // 2,
                validation_data=(X[size // 2 :], y[size // 2 :]),
                validation_batch_size=size // 2,
                epochs=200,
                verbose=False,
            )
        )
    plot_history(histories, path="./plots/multiple_histories.png")
