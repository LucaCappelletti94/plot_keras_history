import matplotlib.pyplot as plt
from typing import List, Dict
import math

def plot_history_graph(axis, history:Dict[str, List[float]], metric:str, run_kind:str):
     axis.plot(history[metric], label='{run_kind} = {value:0.6f}'.format(
            label=metric,
            run_kind=run_kind,
            value=history[metric][-1]))

def get_figsize(n:int, graphs_per_row:int):
    return min(n, graphs_per_row), math.ceil(n/graphs_per_row)

def plot_history(history:Dict[str, List[float]], side:float=5, graphs_per_row:int=4):
    """Plot given training history.
        history:Dict[str, List[float]], the history to plot
        side:int=5, the side of every sub-graph
        graphs_per_row:int=4, number of graphs per row
    """
    metrics = [metric for metric in history if not metric.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n, graphs_per_row)
    _, axes = plt.subplots(h, w, figsize=(side*w,(side-1)*h))

    for axis, metric in zip(axes.flatten(), metrics):
        plot_history_graph(axis, history, metric, "Training")
        testing_metric = "val_{metric}".format(metric=metric)
        if testing_metric in history:
            plot_history_graph(axis, history, testing_metric, "Testing")
        axis.set_title(metric)
        axis.set_xlabel('Epochs')
        axis.legend()
    plt.suptitle("History after {epochs} epochs".format(epochs=len(history["loss"])))