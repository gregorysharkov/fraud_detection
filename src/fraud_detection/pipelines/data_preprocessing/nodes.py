"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.9
"""

import torch_geometric as geom


def download_data() -> geom.data.Data:
    """download and load the Planetoid dataset"""
    dataset = geom.datasets.Planetoid(root='/tmp/Cora', name='Cora')
    return dataset[0]


def train_test_split(dataset) -> tuple[geom.data.Data, geom.data.Data]:
    """Split the dataset into training and test sets"""
    # dataset = dataset.shuffle()
    data = dataset[0]
    train_data = dataset[dataset.train_mask]
    test_data = dataset[dataset.test_mask]
    return train_data, test_data
