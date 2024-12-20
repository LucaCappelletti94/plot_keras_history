"""Utilities for the plot keras history package."""

# type: ignore[no-redef]

import math
from typing import List, Dict, Union, Tuple, cast, TYPE_CHECKING
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

try:
    # We try to import History, but a user may want
    # to plot a CSV without having Keras installed,
    # or more generally, a working version of Keras installed.
    from keras.callbacks import History

    # If the import fails, we create a wrapper class.
except (ImportError, ModuleNotFoundError):
    if not TYPE_CHECKING:
        # pylint: disable=too-few-public-methods
        class History:
            """Dummy class to be used as a placeholder for the real History object."""


def to_dataframe(history: Union[History, pd.DataFrame, Dict, str]) -> pd.DataFrame:
    """Return given history normalized to a dataframe.

    Parameters
    -----------------------------
    history: Union[pd.DataFrame, Dict, str]
        The history object to be normalized.
        Supported values are:
        - pandas DataFrames
        - Dictionaries
        - History object from Keras Callbacks
        - Paths to csv and json files

    Raises
    -----------------------------
    TypeError,
        If given history object is not supported.

    Returns
    -----------------------------
    Normalized pandas dataframe history object.
    """
    if isinstance(history, pd.DataFrame):
        return history
    if isinstance(history, Dict):
        return pd.DataFrame(history)
    if isinstance(history, History):
        return to_dataframe(history.history)
    if isinstance(history, str):
        if "csv" in history.split("."):
            return pd.read_csv(history)
        if "json" in history.split("."):
            expected_dataframe = pd.read_json(history)
            if isinstance(expected_dataframe, pd.DataFrame):
                return expected_dataframe
            if isinstance(expected_dataframe, pd.Series):
                return expected_dataframe.to_frame()

    raise TypeError(
        f"Given history object of type {type(history)} is not currently supported!"
    )


def chain_histories(
    *histories: List[Dict[str, List[float]]],
) -> pd.DataFrame:
    """Return chained histories.

    Parameters
    --------------------
    *histories: List[Dict[str, List[float]]]
        The histories to concate.

    Raises
    --------------------
    ValueError
        If the given histories list is empty.

    Returns
    --------------------
    The concatenated histories.
    """
    if len(histories) == 0:
        raise ValueError("The given histories list is empty!")
    return pd.concat(
        [to_dataframe(history) for history in histories], axis=0
    ).reset_index(drop=True)


def filter_signal(y: np.ndarray, window: int = 17, polyorder: int = 3) -> np.ndarray:
    """Return filtered signal using savgol filter.

    Parameters
    ----------------------------------
    y: List[float]
        The vector to filter.
    window: int = 17
        The size of the window.
        This value MUST be an odd number.
    polyorder: int = 3
        Order of the polynomial.

    Returns
    ----------------------------------
    Filtered vector.
    """
    # The window cannot be smaller than 7 and cannot be greater
    # than the length of the given vector.
    window = max(7, min(window, len(y)))
    # If the window is not odd we force it to be so.
    if window % 2 == 0:
        window -= 1
    # If the window is still bigger than the size of the given vector
    # we return the vector unfiltered.
    if len(y) < window:
        return y
    # Otherwise we apply the savgol filter.
    return cast(np.ndarray, savgol_filter(y, window, polyorder))


def get_figsize(number_of_metrics: int, graphs_per_row: int) -> Tuple[int, int]:
    """Return tuple with the size of the given figures.

    Parameters
    -----------------------------------
    number_of_metrics: int
        Number of the metrics to fit into figure.
    graphs_per_row: int
        Number of graphs to put in each row.


    Returns
    -----------------------------------
    Width and height of the subplots.
    """
    return (
        min(number_of_metrics, graphs_per_row),
        math.ceil(number_of_metrics / graphs_per_row),
    )


def get_column_tuples(history: pd.DataFrame) -> List[List[str]]:
    """Return tuples of the columns to plot.

    Parameters
    -----------------------
    history: pd.DataFrame
        Pandas dataframe with the training history.

    Returns
    -----------------------
    List of the tuples of columns
    """
    return [
        (
            [
                c,
            ]
            if f"val_{c}" not in history
            else [c, f"val_{c}"]
        )
        for c in history.columns
        if not c.startswith("val_") and history[c].notna().all()
    ]


def filter_columns(
    histories: List[pd.DataFrame], columns: List[str]
) -> List[pd.DataFrame]:
    """Return filtered list of dataframes to given columns.

    Parameters
    -----------------------------
    histories: List[pd.DataFrame]
        List of histories as pandas dataframes to filter.
    columns: List[str]
        List of columns to keep.

    Returns
    -----------------------------
    List of filtered history dataframes.
    """
    return [history[columns] for history in histories]
