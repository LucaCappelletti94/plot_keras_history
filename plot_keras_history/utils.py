from typing import List, Dict
import pandas as pd

def to_dataframe(data)->pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data
    return pd.DataFrame(data)

def chain_histories(new_history:Dict[str, List[float]], old_history:Dict[str, List[float]]):
    if old_history is None:
        return to_dataframe(new_history)
    return pd.concat([
        to_dataframe(old_history),
        to_dataframe(new_history)
    ], axis=0).reset_index(drop=True)