"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.9
"""

from functools import partial

import pandas as pd
import torch
import torch_geometric as geom
from torch_geometric.data import Data

from utils.functional_utils import apply_functions, convert_to_tensor

from .edge_utils import add_edge_ids, filter_edges
from .feature_utils import assign_column_names
from .label_utils import get_class_labels


def generate_client_mapper(labels: pd.DataFrame) -> dict[str, str]:
    """maps client IDs to unique IDs for efficient indexing"""

    return {tx_id: idx for idx, tx_id in enumerate(labels["txId"].unique())}


def process_edges(edges: pd.DataFrame, client_mapper: dict[str, str]) -> torch.Tensor:
    """converts edges dataframe to a tensor"""
    return apply_functions(
        df=edges,
        functions=[
            partial(filter_edges, transaction_mapper=client_mapper),
            partial(add_edge_ids, transaction_mapper=client_mapper),
            partial(convert_to_tensor, columns=["left_id", "right_id"]),
        ],
    )


def preprocess_labels(labels: pd.DataFrame) -> torch.Tensor:
    """converts labels dataframe to a tensor"""
    return apply_functions(
        df=labels,
        functions=[
            get_class_labels,
            convert_to_tensor,
        ],
    )


def preprocess_features(features: pd.DataFrame) -> torch.Tensor:
    """converts features dataframe to a tensor"""
    return apply_functions(
        df=features,
        functions=[
            assign_column_names,
            convert_to_tensor,
        ],
    )


def collect_training_data(
    labels: torch.Tensor, edges: torch.Tensor, features: torch.Tensor
) -> geom.data.Data:
    """combines the edge_index, feature_tensor, and node_labels into a single Data object"""
    return Data(edge_index=edges, x=features, y=labels)
