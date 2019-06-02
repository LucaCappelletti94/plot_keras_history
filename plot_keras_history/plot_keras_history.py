import matplotlib.pyplot as plt
from typing import List, Dict
import math
import numpy as np
from .common_labels import common_labels
from scipy.signal import savgol_filter

def get_alias(label: str):
    for name, labels in common_labels.items():
        if label in labels:
            return name
    return label

def filter_signal(y, window:int=17, polyorder=3):
    if len(y)<window:
        return y
    return savgol_filter(y, window, polyorder)

def plot_history_graph(axis, y: List[float], run_kind: str, interpolate:bool):
    axis.plot(filter_signal(y) if interpolate else y, label='{run_kind} = {value:0.6f}'.format(
        run_kind=run_kind,
        value=y[-1]))

def get_figsize(n: int, graphs_per_row: int):
    return min(n, graphs_per_row), math.ceil(n/graphs_per_row)

def plot_history(history: Dict[str, List[float]], interpolate:bool=False, side: float = 5, graphs_per_row: int = 4):
    """Plot given training history.
        history:Dict[str, List[float]], the history to plot.
        interpolate:bool=False, whetever to reduce the graphs noise.
        side:int=5, the side of every sub-graph.
        graphs_per_row:int=4, number of graphs per row.
    """
    metrics = [metric for metric in history if not metric.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n, graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w, (side-1)*h))
    flat_axes = iter(np.array(axes).flatten())


    for metric, axis in zip(metrics, flat_axes):
        plot_history_graph(axis, history[metric], interpolate, "Training")
        testing_metric = "val_{metric}".format(metric=metric)
        if testing_metric in history:
            plot_history_graph(axis, history[testing_metric], interpolate, "Testing")
        axis.set_title(get_alias(metric))
        if n <= graphs_per_row:
            axis.set_xlabel('Epochs')
        epochs = len(history[metric])
        if epochs <= 4:
            axis.set_xticks(np.arange(epochs))
        axis.legend()

    for axis in flat_axes:
        axis.axis("off")

    plt.suptitle("Training history after {epochs} epochs".format(
        epochs=len(history["loss"])))
