import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Callable
import math
import numpy as np
import pandas as pd
from .common_labels import common_labels
from scipy.signal import savgol_filter


def get_alias(label: str):
    for name, labels in common_labels.items():
        if label in labels:
            return name
    return label


def filter_signal(y: List[float], window: int = 17, polyorder: int = 3)->List[float]:
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def get_figsize(n: int, graphs_per_row: int)->Tuple[int, int]:
    return min(n, graphs_per_row), math.ceil(n/graphs_per_row)


def plot_history(history: Union[Dict[str, List[float]], pd.DataFrame], interpolate: bool = False, side: float = 5, graphs_per_row: int = 4, customization_callback: Callable = None):
    """Plot given training history.
        history:Dict[str, List[float]], the history to plot.
        interpolate:bool=False, whetever to reduce the graphs noise.
        side:int=5, the side of every sub-graph.
        graphs_per_row:int=4, number of graphs per row.
        customization_callback:Callable=None, callback for customising axis.
    """
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    x_label = "Epochs" if history.index.name is None else history.index.name
    metrics = [m for m in history if not m.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n, graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w, side*h))
    flat_axes = iter(np.array(axes).flatten())
    for metric, axis in zip(metrics, flat_axes):
        for name, kind in zip((metric, "val_{metric}".format(metric=metric)), ("Train", "Test")):
            if name in history:
                col = history[name]
                axis.plot(
                    col.index,
                    filter_signal(col.values) if interpolate else col.values,
                    label='{kind}: {val:0.4f}'.format(kind=kind, val=col.iloc[-1]))
        alias = get_alias(metric)
        axis.set_xlabel(x_label)
        axis.set_ylabel(alias)
        axis.set_title(alias)
        axis.legend()
        if history.shape[0] <= 4:
            axis.set_xticks(range(history.shape[0]))
        if customization_callback is not None:
            customization_callback(axis)

    for axis in flat_axes:
        axis.axis("off")

    plt.tight_layout()
