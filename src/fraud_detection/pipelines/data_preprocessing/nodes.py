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
            partial(convert_to_tensor, columns=["left_id", "right_id"], tensor_type=torch.int64),
        ],
    )


def preprocess_labels(labels: pd.DataFrame) -> torch.Tensor:
    """converts labels dataframe to a tensor"""
    return apply_functions(
        df=labels,
        functions=[
            get_class_labels,
            partial(convert_to_tensor, tensor_type=torch.long),
        ],
        debug=True,
    )


def preprocess_features(features: pd.DataFrame) -> torch.Tensor:
    """converts features dataframe to a tensor"""
    return apply_functions(
        df=features,
        functions=[
            assign_column_names,
            partial(convert_to_tensor, tensor_type=torch.float),
        ],
    )


def collect_training_data(
    labels: torch.Tensor,
    edges: torch.Tensor,
    features: torch.Tensor,
    train_ratio: float = 0.8,
    test_ratio: float = 0.1,
) -> geom.data.Data:
    """combines the edge_index, feature_tensor, and node_labels into a single Data object"""
    data = Data(edge_index=edges, x=features, y=labels)
    data = train_test_split(data, train_ratio, test_ratio)
    return data


def train_test_split(
    data: geom.data.Data, train_ratio: float, test_ratio: float
) -> geom.data.Data:
    """sets train, test and validation masks for the data"""
    known_mask = (data.y == 0) | (data.y == 1)

    number_of_known_nodes = known_mask.sum().item()
    permutation = torch.randperm(number_of_known_nodes)
    train_size = int(train_ratio * number_of_known_nodes)
    val_size = int(test_ratio * number_of_known_nodes)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_indices = known_mask.nonzero(as_tuple=True)[0][permutation[:train_size]]
    train_mask[train_indices] = True
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_indices = known_mask.nonzero(as_tuple=True)[0][
        permutation[train_size : train_size + val_size]
    ]
    val_mask[val_indices] = True
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_indices = known_mask.nonzero(as_tuple=True)[0][
        permutation[train_size + val_size :]
    ]
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.known_mask = known_mask

    return data
