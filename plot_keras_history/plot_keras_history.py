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
    average_history: pd.DataFrame = None,
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Callable = None,
    path: str = None,
    max_epochs: int = None,
    log_scale_metrics: bool = False,
    monitor: str = None,
    best_point_x: int = None,
    custom_defaults: Dict[str, Union[List[str], str]] = None
):
    """Plot given training histories.

    Parameters
    -------------------------------
    histories:pd.DataFrame,
        The histories to plot.
    average_history: pd.DataFrame = None,
        Average histories, if multiple histories were given.
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
    monitor: str = None,
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    best_point_x: int = None,
        Point to be highlighted as best.
    custom_defaults: Dict[str, Union[List[str], str]] = None,
        Dictionary of custom mapping to use to sanitize metric names.
    """
    x_label = "Epochs" if histories[0].index.name is None else histories[0].index.name
    metrics = [m for m in histories[0] if not m.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n, graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w, side*h))
    flat_axes = np.array(axes).flatten()

    for i, history in enumerate([average_history] + histories):
        for metric, axis in zip(metrics, flat_axes):
            for name, kind in zip(
                *(
                    ((metric, f"val_{metric}"), ("Train", "Test"))
                    if f"val_{metric}" in history
                    else ((metric, ), ("", ))
                )
            ):
                col = history[name]
                if i == 0:
                    if best_point_x is not None:
                        best_point_y = col.values[best_point_x]
                        if len(kind) == 0:
                            kind = f"Best value ({monitor})"
                        else:
                            kind = f"{kind} best value ({monitor})"
                    else:
                        best_point_y = col.iloc[-1]
                        if len(kind) == 0:
                            kind = f"Last value"
                        else:
                            kind = f"{kind} last value"

                    line = axis.plot(
                        col.index.values,
                        filter_signal(
                            col.values) if interpolate else col.values,
                        style,
                        label='{kind}: {val:0.4f}'.format(
                            kind=kind,
                            val=best_point_y
                        ),
                        zorder=10000
                    )[0]
                    if best_point_x is not None:
                        best_point_y = col.values[best_point_x]
                        axis.scatter(
                            [best_point_x],
                            [best_point_y],
                            s=30,
                            alpha=0.9,
                            color=line.get_color(),
                            zorder=10000
                        )
                        axis.hlines(
                            best_point_y,
                            0,
                            best_point_x,
                            linestyles="dashed",
                            color=line.get_color(),
                            alpha=0.5,
                        )
                        axis.vlines(
                            best_point_x,
                            0,
                            best_point_y,
                            linestyles="dashed",
                            color=line.get_color(),
                            alpha=0.5,
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
        alias = sanitize_ml_labels(metric, custom_defaults=custom_defaults)
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
    return [
        [c]
        if f"val_{c}" not in history
        else [c,  f"val_{c}"]
        for c in history.columns
        if not c.startswith("val_") and history[c].notna().all()
    ]


def filter_column(histories: List[str], columns: List[str]) -> List[pd.DataFrame]:
    return [history[columns] for history in histories]


def to_dataframe(history) -> pd.DataFrame:
    if isinstance(history, pd.DataFrame):
        return history
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
    monitor: str = None,
    monitor_mode: str = "max",
    log_scale_metrics: bool = False,
    custom_defaults: Dict[str, Union[List[str], str]] = None
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
    monitor: str = None,
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    monitor_mode: str = "max",
        Mode to display the monitor metric best point.
        Can either be "max" or "min".
    log_scale_metrics: bool = False,
        Wether to use log scale for the metrics.
    custom_defaults: Dict[str, Union[List[str], str]] = None,
        Dictionary of custom mapping to use to sanitize metric names.

    Raises
    --------------------------
    ValueError,
        Currently the monitor metric best point cannot be displayed if interpolation is active.
    ValueError,
        If monitor_mode is not either "min" or "max".
    ValueError,
        If max_epochs is not either "min", "max" or a numeric integer.
    """
    if interpolate and monitor is not None:
        raise ValueError((
            "Currently the monitor metric best point "
            "cannot be displayed if interpolation is active."
        ))
    if monitor_mode not in ("min", "max"):
        raise ValueError("Given monitor mode '{}' is not supported.".format(
            monitor_mode
        ))
    if max_epochs not in ("min", "max") and not isinstance(max_epochs, int):
        raise ValueError("Given parameter max_epochs '{}' is not supported.".format(
            max_epochs
        ))
    if not isinstance(histories, list):
        histories = [histories]
    if path is not None:
        dirname = os.path.dirname(path)
        if dirname:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    histories = [
        to_dataframe(history)._get_numeric_data()
        for history in histories
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

    if len(histories) > 0:
        average_history = pd.concat(histories)
        average_history = average_history.groupby(average_history.index).mean()
    else:
        average_history = None

    if monitor is not None:
        history_to_monitor = (
            histories[0] if average_history is None else average_history)[monitor]
        if monitor_mode == "max":
            best_point_x = history_to_monitor.argmax()
        elif monitor_mode == "min":
            best_point_x = history_to_monitor.argmin()

    else:
        best_point_x = None

    if single_graphs:
        for columns in _get_columns(histories[0]):
            _plot_history(
                filter_column(histories, columns),
                average_history,
                style,
                interpolate,
                side,
                graphs_per_row,
                customization_callback,
                "{path}/{c}.png".format(path=path, c=columns[0]),
                log_scale_metrics,
                monitor=sanitize_ml_labels(
                    monitor,
                    custom_defaults=custom_defaults
                ),
                best_point_x=best_point_x,
                custom_defaults=custom_defaults
            )
    else:
        _plot_history(
            histories,
            average_history,
            style,
            interpolate,
            side,
            graphs_per_row,
            customization_callback,
            path,
            log_scale_metrics,
            monitor=sanitize_ml_labels(
                monitor,
                custom_defaults=custom_defaults
            ),
            best_point_x=best_point_x,
            custom_defaults=custom_defaults
        )
