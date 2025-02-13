import logging
from functools import reduce
from typing import Callable

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def apply_functions(
    df: pd.DataFrame, functions: list[Callable], debug: bool = False
) -> pd.DataFrame:
    """applies a list of functions to a dataframe"""
    if not debug:
        return reduce(lambda df, func: func(df), functions, df)

    for func in functions:
        df = func(df)
        logger.debug(f"df after transformation:\n\t{df[:10]=}\n\t{df.shape=}")

    return df


def convert_to_tensor(
    data: pd.DataFrame, columns: list[str] = None, tensor_type=torch.float
) -> torch.Tensor:
    """converts the edges dataframe to a tensor"""

    if columns:
        data = data[[*columns]].values.T

    if isinstance(data, pd.DataFrame):
        data = data.values

    return torch.tensor(data, dtype=tensor_type)
