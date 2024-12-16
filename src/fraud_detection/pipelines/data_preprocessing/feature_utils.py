import pandas as pd
import torch


def assign_column_names(features: pd.DataFrame) -> torch.Tensor:
    """preprocess the features dataframe"""

    features.columns = ["txId"] + [f"F{i}" for i in range(1, len(features.columns))]
    features.drop(columns=["txId"], inplace=True)
    return features
