from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import numpy as np
import os
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def test_plotting_keras_history_object():
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
