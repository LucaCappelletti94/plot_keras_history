import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Callable
import os
import math
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sanitize_ml_labels import sanitize_ml_labels, is_normalized_metric


def filter_signal(y: List[float], window: int = 17, polyorder: int = 3) -> List[float]:
    window = max(7, min(window, len(y)))
    if window % 2 == 0:
        window -= 1
    if len(y) < window:
        return y
    return savgol_filter(y, window, polyorder)


def get_figsize(n: int, graphs_per_row: int) -> Tuple[int, int]:
    return min(n, graphs_per_row), math.ceil(n/graphs_per_row)


def _plot_history(
    histories: pd.DataFrame,
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Callable = None,
    path: str = None,
    max_epochs: int = None,
    log_scale_metrics: bool = False
):
    """Plot given training histories.

    Parameters
    -------------------------------
    histories:pd.DataFrame,
        The histories to plot.
    style:str="-",
        The style to use when plotting the graphs.
    interpolate:bool=False,
        Whetever to reduce the graphs noise.
    side:int=5,
        The side of every sub-graph.
    graphs_per_row:int=4,
        Number of graphs per row.
    customization_callback:Callable=None,
        Callback for customising axis.
    path:str=None,
        Where to save the graphs, by defalut nowhere.
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    """
    x_label = "Epochs" if histories[0].index.name is None else histories[0].index.name
    metrics = [m for m in histories[0] if not m.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n, graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w, side*h))
    flat_axes = np.array(axes).flatten()

    average_history = pd.concat(histories)
    average_history = average_history.groupby(average_history.index).mean()

    for i, history in enumerate([average_history] + histories):
        for metric, axis in zip(metrics, flat_axes):
            for name, kind in zip((metric, f"val_{metric}"), ("Train", "Test")):
                if name in history:
                    col = history[name]
                    if i == 0:
                        axis.plot(
                            col.index.values,
                            filter_signal(
                                col.values) if interpolate else col.values,
                            style,
                            label='{kind}: {val:0.4f}'.format(
                                kind=kind, val=col.iloc[-1]),
                            zorder=10000
                        )
                    else:
                        axis.plot(
                            col.index.values,
                            filter_signal(
                                col.values) if interpolate else col.values,
                            style,
                            alpha=0.3
                        )

    for metric, axis in zip(metrics, flat_axes):
        alias = sanitize_ml_labels(metric)
        axis.set_xlabel(x_label)
        if log_scale_metrics:
            axis.set_yscale("log")
        axis.set_ylabel("{alias}{scale}".format(
            alias=alias,
            scale=" (Log scale)" if log_scale_metrics else ""
        ))
        axis.set_title(alias)
        axis.grid(True)
        axis.legend()
        if is_normalized_metric(metric):
            axis.set_ylim(-0.05, 1.05)
        if history.shape[0] <= 4:
            axis.set_xticks(range(history.shape[0]))
        if customization_callback is not None:
            customization_callback(axis)

    for axis in flat_axes[len(metrics):]:
        axis.axis("off")

    plt.tight_layout()
    if path is not None:
        plt.savefig(path)


def _get_columns(history: pd.DataFrame) -> List[str]:
    return [[c] if f"val_{c}" not in history else [c,  f"val_{c}"] for c in history if not c.startswith("val_")]


def filter_column(histories: List[str], columns: List[str]) -> List[pd.DataFrame]:
    return [history[columns] for history in histories]


def to_dataframe(history):
    if isinstance(history, Dict):
        return pd.DataFrame(history)
    if isinstance(history, str):
        if "csv" in history.split("."):
            return pd.read_csv(history)
        if "json" in history.split("."):
            return pd.read_json(history)
    raise ValueError("Invalid history term of type {history_type}".format(
        history_type=type(history)
    ))


def plot_history(
    histories: Union[Dict[str, List[float]], pd.DataFrame, List[pd.DataFrame], str, List[str]],
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Callable = None,
    path: str = None,
    single_graphs: bool = False,
    max_epochs: Union[int, str] = "max",
    log_scale_metrics: bool = False
):
    """Plot given training histories.

    Parameters
    ----------------------------
    histories,
        the histories to plot.
        This parameter can either be a single or multiple dataframes
        or one or more paths to the stored CSVs or JSON of the history.
    style:str="-",
        the style to use when plotting the graphs.
    interpolate:bool=False,
        whetever to reduce the graphs noise.
    side:int=5,
        the side of every sub-graph.
    graphs_per_row:int=4,
        number of graphs per row.
    customization_callback:Callable=None,
        callback for customising axis.
    path:str=None,
        where to save the graphs, by defalut nowhere.
    single_graphs:bool=False,
        whetever to create the graphs one by one.
    max_epochs: Union[int, str] = "max",
        Number of epochs to plot. Can either be "max", "min" or a positive integer value.
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    """
    if not isinstance(histories, list):
        histories = [histories]
    if path is not None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(os.path.dirname(path), exist_ok=True)
    histories = [
        to_dataframe(history) if not isinstance(history, pd.DataFrame) else history for history in histories
    ]
    if max_epochs in ("max", "min"):
        epochs = [
            len(history)
            for history in histories
        ]
        if max_epochs == "max":
            max_epochs = max(epochs)
        if max_epochs == "min":
            max_epochs = min(epochs)

    histories = [
        history[:max_epochs]
        for history in histories
    ]

    if single_graphs:
        for columns in _get_columns(histories[0]):
            _plot_history(
                filter_column(histories, columns),
                style,
                interpolate,
                side,
                graphs_per_row,
                customization_callback,
                "{path}/{c}.png".format(path=path, c=columns[0]),
                log_scale_metrics
            )
    else:
        _plot_history(histories, style, interpolate, side, graphs_per_row,
                      customization_callback, path, log_scale_metrics)
