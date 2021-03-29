from typing import List, Dict, Union
import pandas as pd


def to_dataframe(history: Union[pd.DataFrame, Dict, str]) -> pd.DataFrame:
    """Return given history normalized to a dataframe.

    Parameters
    -----------------------------
    history: Union[pd.DataFrame, Dict, str],
        The history object to be normalized.
        Supported values are:
        - pandas DataFrames
        - Dictionaries
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
    if isinstance(history, str):
        if "csv" in history.split("."):
            return pd.read_csv(history)
        if "json" in history.split("."):
            return pd.read_json(history)
    raise TypeError("Given history object of type {history_type} is not currently supported!".format(
        history_type=type(history)
    ))


def chain_histories(new_history: Dict[str, List[float]], old_history: Dict[str, List[float]]):
    if old_history is None:
        return to_dataframe(new_history)
    return pd.concat([
        to_dataframe(old_history),
        to_dataframe(new_history)
    ], axis=0).reset_index(drop=True)
