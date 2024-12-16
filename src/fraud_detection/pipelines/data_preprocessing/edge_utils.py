
import pandas as pd
import torch


def filter_edges(edges: pd.DataFrame, transaction_mapper: dict) -> pd.DataFrame:
    """removes all edges that do not have features in the features dataframe"""

    edge_ids = set(transaction_mapper.keys())
    left_condition = edges.txId1.isin(edge_ids)
    right_condition = edges.txId2.isin(edge_ids)
    return edges[left_condition & right_condition]


def add_edge_ids(edges: pd.DataFrame, transaction_mapper: dict) -> pd.DataFrame:
    """adds edge ids to the edges dataframe"""
    edges["left_id"] = edges.txId1.map(transaction_mapper)
    edges["right_id"] = edges.txId2.map(transaction_mapper)

    return edges

