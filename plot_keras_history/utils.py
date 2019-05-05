from typing import List, Dict

def chain_histories(new_history:Dict[str, List[float]], old_history:Dict[str, List[float]]):
    if old_history is None:
        return new_history
    return {
        key: old_history[key] + new_history[key] for key in old_history
    }