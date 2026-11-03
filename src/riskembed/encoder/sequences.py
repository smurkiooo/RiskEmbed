"""Shared conversion between timestamps and integer day indices."""

import numpy as np
import pandas as pd

EPOCH = np.datetime64("1970-01-01", "D")

def day_number(dates: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    return (pd.DatetimeIndex(dates).to_numpy("datetime64[D]") - EPOCH).astype(np.int32)
