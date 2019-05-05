import matplotlib.pyplot as plt
from typing import List, Dict
import math

def plot_history_graph(axis, history:Dict[str, List[float]], metric:str, run_kind:str):
     axis.plot(history[metric], label='{run_kind} = {value:0.6f}'.format(
            label=metric,
            run_kind=run_kind,
            value=history[metric][-1]))

def get_figsize(n:int, _max:int=4):
    return min(n, _max), math.ceil(n/_max)

def plot_history(history:Dict[str, List[float]], side:int=5):
    metrics = [metric for metric in history if not metric.startswith("val_")]
    n = len(metrics)
    w, h = get_figsize(n)
    _, axes = plt.subplots(h, n, figsize=(side*w,(side-1)*h))
    for axis, metric in zip(axes, metrics):
        plot_history_graph(axis, history, metric, "Training")
        plot_history_graph(axis, history, "val_{metric}".format(metric=metric), "Testing")
        axis.set_title(metric)
        axis.set_ylabel("Value")
        axis.set_xlabel('Epochs')
        axis.legend()
