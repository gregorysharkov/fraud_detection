from functools import reduce
from typing import Callable

import pandas as pd
import torch


def apply_functions (df: pd.DataFrame, functions: list[Callable]) -> pd.DataFrame:
    """applies a list of functions to a dataframe"""
    return reduce(lambda df, func: func(df), functions, df)


def convert_to_tensor(data: pd.DataFrame, columns: list[str] = None) -> torch.Tensor:
    """converts the edges dataframe to a tensor"""

    if columns:
        data = data[[*columns]].values.T

    if isinstance(data, pd.DataFrame):
        data = data.values

    return torch.tensor(data, dtype=torch.long)
