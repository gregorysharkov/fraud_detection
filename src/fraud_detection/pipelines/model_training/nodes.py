"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.9
"""

import logging

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as geom
from sklearn import metrics
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
    input_data: geom.data.Data,
    train_params: dict,
    device: str,
) -> tuple[torch.nn.Module, dict, dict]:
    """
    trains a model given training_parameters

    Args:
        train_data (geom.data.Data): training dataset
        train_params (dict): training parameters
        device (str): device to use ("gpu", or "mps", or "cpu")
    """
    logger.info(f"{input_data=}")

    input_data = input_data.to(device)
    model = GCN(
        num_features=input_data.num_node_features,
        hidden_size=train_params["hidden_size"],
        num_classes=torch.unique(input_data.y).numel(),
    ).to(device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["learning_rate"])

    train_metrics = {
        "train_loss": [],
        "train_accuracy": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
    }

    val_metrics = {
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
    }
    for _epoch in tqdm(range(train_params["num_epochs"])):
        model.train()
        optimizer.zero_grad()
        out = model(input_data)
        predicted_y = out[input_data.train_mask]
        true_y = input_data.y[input_data.train_mask]
        print(f"predicted_y: {predicted_y.shape}, true_y: {true_y.shape}")
        print(f"predicted_y shape: {predicted_y.shape}, dtype: {predicted_y.dtype}")
        print(f"true_y shape: {true_y.shape}, dtype: {true_y.dtype}")
        loss = F.nll_loss(predicted_y, true_y)
        loss.backward()
        optimizer.step()
        print("Epoch: {}, Train loss: {:.4f}".format(_epoch, loss.item()))

        train_metrics = _evaluate_model(
            model, input_data.cpu(), input_data.train_mask.cpu(), train_metrics
        )
        # val_metrics = _evaluate_model(
        #     model, input_data.cpu(), input_data.val_mask.cpu(), val_metrics
        # )

    return model, train_metrics, val_metrics


def _evaluate_model(
    model: torch.nn.Module,
    data: geom.data.Data,
    data_mask: list[bool],
    current_metrics: dict,
) -> dict[str, list[float]]:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        data_mask = data_mask.to(device)
        model_out = model(data)
        eval_y = data.y[data_mask].argmax(dim=1)
        pred_y = model_out[data_mask]
        loss = F.nll_loss(pred_y, eval_y)

        # Move tensors to CPU for metric calculation
        pred_y = pred_y.cpu()
        eval_y = eval_y.cpu()

        current_metrics = _update_metrics(
            predictions=pred_y.numpy(),
            true_values=eval_y.numpy(),
            current_metrics=current_metrics,
            loss=loss.item(),
        )

    return current_metrics


def _update_metrics(
    predictions: np.ndarray,
    true_values: torch.Tensor,
    current_metrics: dict[list[float]],
    loss: float,
) -> list[float]:
    """Evaluates model performance and updates the metrics dictionary"""

    accuracy = metrics.accuracy_score(true_values, predictions)
    precision = metrics.precision_score(
        true_values, predictions, average="weighted"
    ).item()
    recall = metrics.recall_score(true_values, predictions, average="weighted").item()
    f1_score = metrics.f1_score(true_values, predictions, average="weighted").item()

    current_metrics["train_loss"].append(loss)
    # current_metrics["train_accuracy"].append(accuracy)
    # current_metrics["train_precision"].append(precision)
    # current_metrics["train_recall"].append(recall)
    # current_metrics["train_f1"].append(f1_score)

    return current_metrics
