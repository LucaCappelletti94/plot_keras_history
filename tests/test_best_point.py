import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from plot_keras_history import plot_history

matplotlib.use("Agg")


def test_monitor():
    plot_history(
        [
            pd.read_csv("tests/history1.csv"),
            pd.read_csv("tests/history2.csv"),
            pd.read_csv("tests/history4.csv"),
            pd.read_csv("tests/history3.csv"),
        ],
        path="plots/monitor.png",
        monitor="val_auprc",
        monitor_mode="max",
        custom_defaults={"Test AUPRC": "val_auprc"},
    )
    plot_history(
        [
            pd.read_csv("tests/history1.csv"),
            pd.read_csv("tests/history2.csv"),
            pd.read_csv("tests/history4.csv"),
            pd.read_csv("tests/history3.csv"),
        ],
        path="plots/monitor.png",
        monitor="val_auprc",
        monitor_mode="min",
        custom_defaults={"Test AUPRC": "val_auprc"},
    )
    plt.close()
