"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.9
"""
import logging

import torch
import torch.nn.functional as F
import torch_geometric as geom
from tqdm import tqdm

from .gnn import GCN

DEVICE_MAPPING = {
    "gpu": torch.cuda,
    "mps": torch.backends.mps,
}

logger = logging.getLogger(__name__)


def get_torch_device() -> torch.device:
    """gets the best performant device (CPU or MPS)"""
    for device_name, device in DEVICE_MAPPING.items():
        if device.is_available():
            logger.info(f"Using {device_name} device")
            return torch.device(device_name)

    logger.info(f"Using {device_name} device")
    return torch.device("cpu")


def train_model(
    train_data: geom.data.Data,
    train_params: dict,
    device: str,
) -> torch.nn.Module:
    """
    trains a model given training_parameters

    Args:
        train_data (geom.data.Data): training dataset
        train_params (dict): training parameters
        device (str): device to use ("gpu", or "mps", or cpu)
    """
    train_data = train_data.to(device)
    model = GCN(
        num_features=train_data.num_node_features,
        hidden_size=train_params["hidden_size"],
        num_classes=torch.unique(train_data.y).numel(),
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["learning_rate"])

    for epoch in tqdm(range(train_params["num_epochs"])):
        optimizer.zero_grad()
        out = model(train_data)
        train_loss = F.nll_loss(
            out[train_data.train_mask], train_data.y[train_data.train_mask]
        )
        test_loss = F.nll_loss(
            out[train_data.test_mask], train_data.y[train_data.test_mask]
        )
        if epoch % 10 == 0:  # print every 10th epoch
            logger.info(
                f"Epoch: {epoch}, Train loss: {train_loss.item()}, Test loss: {test_loss.item()}"
            )
        train_loss.backward()
        optimizer.step()

    return model
