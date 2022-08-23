"""Methods for plotting a keras model training history."""
import warnings
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Tuple, Callable, Optional
import os
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from sanitize_ml_labels import sanitize_ml_labels, is_normalized_metric, is_absolutely_normalized_metric
from .utils import to_dataframe, get_figsize, filter_signal, get_column_tuples, filter_columns, History


def _plot_history(
    histories: List[pd.DataFrame],
    average_history: Optional[pd.DataFrame] = None,
    standard_deviation_history: Optional[pd.DataFrame] = None,
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Optional[Callable] = None,
    path: Optional[str] = None,
    log_scale_metrics: bool = False,
    show_standard_deviation: bool = False,
    show_average: bool = True,
    monitor: Optional[str] = None,
    best_point_x: Optional[int] = None,
    title: Optional[str] = None,
    custom_defaults: Optional[Dict[str, Union[List[str], str]]] = None
) -> Tuple[Figure, Axes]:
    """Plot given training histories.

    Parameters
    -------------------------------
    histories: List[pd.DataFrame]
        The histories to plot.
    average_history: pd.DataFrame = None
        Average histories, if multiple histories were given.
    standard_deviation_history: Optional[pd.DataFrame] = None
        Standard deviation histories, if multiple histories were given.
    style: str = "-"
        The style to use when plotting the graphs.
    interpolate: bool = False
        Whetever to reduce the graphs noise.
    side: int = 5
        The side of every sub-graph.
    graphs_per_row: int = 4
        Number of graphs per row.
    customization_callback: Callable = None
        Callback for customising axis.
    path:str = None
        Where to save the graphs, by defalut nowhere.
    monitor: str = None
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    log_scale_metrics: bool = False
        Whether to use log scale for the metrics.
    show_standard_deviation: bool = False
        Whether to show the standard deviation when
        plotting multiple training histories.
    show_average: bool = True
        Whether to show the average when
        plotting multiple training histories.
    best_point_x: int = None
        Point to be highlighted as best.
    title: str = None
        Title to put on top of the subplots.
    custom_defaults: Dict[str, Union[List[str], str]] = None
        Dictionary of custom mapping to use to sanitize metric names.
    """
    x_label = "Epochs" if histories[0].index.name is None else histories[0].index.name
    metrics = [
        c[0]
        for c in get_column_tuples(histories[0])
    ]
    number_of_metrics = len(metrics)
    w, h = get_figsize(number_of_metrics, graphs_per_row)
    fig, axes = plt.subplots(h, w, figsize=(
        side*w, side*h), constrained_layout=True)
    flat_axes = np.array(axes).flatten()

    if show_average and average_history is not None:
        histories = [average_history] + histories

    for i, history in enumerate(histories):
        for metric, axis in zip(metrics, flat_axes):
            for name, kind, color in zip(
                *(
                    ((metric, f"val_{metric}"), ("Train", "Test"), ("tab:blue", "tab:orange"))
                    if f"val_{metric}" in history
                    else ((metric, ), ("", ), ("tab:blue",))
                )
            ):
                col = history[name]
                if is_normalized_metric(metric):
                    min_value = col.values.min()
                    max_value = col.values.max()
                    if min_value < 0.0 or max_value > 1.0:
                        warnings.warn(
                            (
                                "Please be advised that you have provided a metric called `{metric}` "
                                "that is expected to be normalized, i.e. between 0 and 1. The values "
                                "you have provided for this metric were between {min_value:0.3f} and "
                                "{max_value:0.3f}."
                            ).format(
                                metric=metric,
                                min_value=min_value,
                                max_value=max_value
                            )
                        )
                if is_absolutely_normalized_metric(metric):
                    min_value = col.values.min()
                    max_value = col.values.max()
                    if min_value < -1.0 or max_value > 1.0:
                        warnings.warn(
                            (
                                "Please be advised that you have provided a metric called `{metric}` "
                                "that is expected to be absolutely normalized, i.e. between -1 and 1. The values "
                                "you have provided for this metric were between {min_value:0.3f} and "
                                "{max_value:0.3f}."
                            ).format(
                                metric=metric,
                                min_value=min_value,
                                max_value=max_value
                            )
                        )
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

                    values = filter_signal(
                        col.values
                    ) if interpolate else col.values

                    if show_standard_deviation and standard_deviation_history is not None:
                        axis.fill_between(
                            col.index.values,
                            values-standard_deviation_history[name].values,
                            values+standard_deviation_history[name].values,
                            color=color,
                            alpha=0.1
                        )
                        axis.plot(
                            col.index.values,
                            values-standard_deviation_history[name].values,
                            color=color,
                            linewidth=0.5,
                            alpha=0.1
                        )
                        axis.plot(
                            col.index.values,
                            values+standard_deviation_history[name].values,
                            color=color,
                            linewidth=0.5,
                            alpha=0.1
                        )
                    line = axis.plot(
                        col.index.values,
                        values,
                        style,
                        label='{kind}: {val:0.4f}'.format(
                            kind=kind,
                            val=best_point_y
                        ),
                        linewidth=2 if len(histories) > 1 else 1,
                        color=color,
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
                            alpha=0.6,
                        )
                else:
                    axis.plot(
                        col.index.values,
                        filter_signal(
                            col.values) if interpolate else col.values,
                        style,
                        color=color,
                        alpha=0.5
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
        elif is_absolutely_normalized_metric(metric):
            axis.set_ylim(-1.05, 1.05)
        if history.shape[0] <= 4:
            axis.set_xticks(range(history.shape[0]))
        if customization_callback is not None:
            customization_callback(axis)

    for axis in flat_axes[len(metrics):]:
        axis.axis("off")

    if title is not None:
        fig.suptitle(title, fontsize=20)

    if path is not None:
        fig.savefig(path)

    return fig, axes


def plot_history(
    histories: Union[History, List[History], Dict[str, List[float]], pd.DataFrame, List[pd.DataFrame], str, List[str]],
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Optional[Callable] = None,
    path: Optional[str] = None,
    single_graphs: bool = False,
    max_epochs: Union[int, str] = "max",
    monitor: Optional[str] = None,
    monitor_mode: str = "max",
    log_scale_metrics: bool = False,
    show_standard_deviation: bool = False,
    show_average: bool = True,
    title: Optional[str] = None,
    custom_defaults: Optional[Dict[str, Union[List[str], str]]] = None
) -> Tuple[Union[Figure, List[Figure]], Union[Axes, List[Axes]]]:
    """Plot given training histories.

    Parameters
    ----------------------------
    histories
        the histories to plot.
        This parameter can either be a single or multiple dataframes
        or one or more paths to the stored CSVs or JSON of the history.
    style: str = "-"
        the style to use when plotting the graphs.
    interpolate: bool = False
        whetever to reduce the graphs noise.
    side: int = 5
        the side of every sub-graph.
    graphs_per_row: int = 4
        number of graphs per row.
    customization_callback: Callable = None
        callback for customising axis.
    path: str = None
        where to save the graphs, by defalut nowhere.
    single_graphs: bool = False
        whetever to create the graphs one by one.
    max_epochs: Union[int, str] = "max"
        Number of epochs to plot. Can either be "max", "min" or a positive integer value.
    monitor: str = None
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    monitor_mode: str = "max"
        Mode to display the monitor metric best point.
        Can either be "max" or "min".
    log_scale_metrics: bool = False
        Whether to use log scale for the metrics.
    show_standard_deviation: bool = False
        Whether to show the standard deviation when
        plotting multiple training histories.
    show_average: bool = True
        Whether to show the average when
        plotting multiple training histories.
    title: str = None,
        Title to put on top of the subplots.
    custom_defaults: Dict[str, Union[List[str], str]] = None
        Dictionary of custom mapping to use to sanitize metric names.

    Raises
    --------------------------
    ValueError
        Currently the monitor metric best point cannot be displayed if interpolation is active.
    ValueError
        If monitor_mode is not either "min" or "max".
    ValueError
        If max_epochs is not either "min", "max" or a numeric integer.
    """
    # Some parameters validation
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
    # If the histories are not provided as a list, we normalized it
    # to a list.
    if not isinstance(histories, list):
        histories = [histories]
    # If the path is not None, we prepare the directory where to
    # store the created image(s).
    if path is not None:
        directory_name = os.path.dirname(path)
        # The directory name may be an empty string.
        if directory_name:
            os.makedirs(directory_name, exist_ok=True)

    # Normalize the training histories.
    histories = [
        to_dataframe(history)._get_numeric_data()
        for history in histories
    ]

    # Filter out the epochs as required.
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

    if len(histories) > 1:
        grouped_histories = pd.concat(histories)
        average_history = grouped_histories.groupby(grouped_histories.index).mean()
        standard_deviation_history = grouped_histories.groupby(grouped_histories.index).std()
    else:
        average_history = standard_deviation_history =  None

    # If we want to plot informations relative to the monitored metrics
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
        return list(zip(*[
            _plot_history(
                filter_columns(histories, columns),
                average_history,
                standard_deviation_history,
                style,
                interpolate,
                side,
                graphs_per_row,
                customization_callback,
                path = None if path is None else "{path}/{c}.png".format(path=path, c=columns[0]),
                log_scale_metrics=log_scale_metrics,
                show_standard_deviation=show_standard_deviation,
                monitor=sanitize_ml_labels(
                    monitor,
                    custom_defaults=custom_defaults
                ),
                best_point_x=best_point_x,
                title=title,
                custom_defaults=custom_defaults,
            )
            for columns in get_column_tuples(histories[0])
        ]))
    else:
        return _plot_history(
            histories,
            average_history,
            standard_deviation_history,
            style,
            interpolate,
            side,
            graphs_per_row,
            customization_callback,
            path,
            log_scale_metrics=log_scale_metrics,
            show_standard_deviation=show_standard_deviation,
            show_average=show_average,
            monitor=sanitize_ml_labels(
                monitor,
                custom_defaults=custom_defaults
            ),
            best_point_x=best_point_x,
            title=title,
            custom_defaults=custom_defaults,
        )


def show_history(
    histories: Union[History, List[History], Dict[str, List[float]], pd.DataFrame, List[pd.DataFrame], str, List[str]],
    style: str = "-",
    interpolate: bool = False,
    side: float = 5,
    graphs_per_row: int = 4,
    customization_callback: Optional[Callable] = None,
    path: Optional[str] = None,
    single_graphs: bool = False,
    max_epochs: Union[int, str] = "max",
    monitor: Optional[str] = None,
    monitor_mode: str = "max",
    log_scale_metrics: bool = False,
    show_standard_deviation: bool = False,
    show_average: bool = True,
    title: Optional[str] = None,
    custom_defaults: Optional[Dict[str, Union[List[str], str]]] = None
) -> Tuple[Union[Figure, List[Figure]], Union[Axes, List[Axes]]]:
    """Plot given training histories.

    Parameters
    ----------------------------
    histories
        the histories to plot.
        This parameter can either be a single or multiple dataframes
        or one or more paths to the stored CSVs or JSON of the history.
    style: str = "-"
        the style to use when plotting the graphs.
    interpolate: bool = False
        whetever to reduce the graphs noise.
    side: int = 5
        the side of every sub-graph.
    graphs_per_row: int = 4
        number of graphs per row.
    customization_callback: Callable = None
        callback for customising axis.
    path: str = None
        where to save the graphs, by defalut nowhere.
    single_graphs: bool = False
        whetever to create the graphs one by one.
    max_epochs: Union[int, str] = "max"
        Number of epochs to plot. Can either be "max", "min" or a positive integer value.
    monitor: str = None
        Metric to use to display best points.
        For example you may use "loss" or "val_loss".
        By default None, to not display any best point.
    monitor_mode: str = "max"
        Mode to display the monitor metric best point.
        Can either be "max" or "min".
    log_scale_metrics: bool = False
        Whether to use log scale for the metrics.
    show_standard_deviation: bool = False
        Whether to show the standard deviation when
        plotting multiple training histories.
    show_average: bool = True
        Whether to show the average when
        plotting multiple training histories.
    title: str = None
        Title to put on top of the subplots.
    custom_defaults: Dict[str, Union[List[str], str]] = None
        Dictionary of custom mapping to use to sanitize metric names.

    Raises
    --------------------------
    ValueError
        Currently the monitor metric best point cannot be displayed if interpolation is active.
    ValueError
        If monitor_mode is not either "min" or "max".
    ValueError
        If max_epochs is not either "min", "max" or a numeric integer.
    """
    plot_history(
        histories=histories,
        style=style,
        interpolate=interpolate,
        side=side,
        graphs_per_row=graphs_per_row,
        customization_callback=customization_callback,
        path=path,
        single_graphs=single_graphs,
        max_epochs=max_epochs,
        monitor=monitor,
        monitor_mode=monitor_mode,
        log_scale_metrics=log_scale_metrics,
        show_standard_deviation=show_standard_deviation,
        show_average=show_average,
        title=title,
        custom_defaults=custom_defaults,
    )
    plt.show()
