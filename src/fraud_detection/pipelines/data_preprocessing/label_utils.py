import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder


def get_class_labels(labels: pd.DataFrame) -> torch.Tensor:
    """converts the labels dataframe to a tensor"""
    label_encoder = LabelEncoder()
    class_labels = label_encoder.fit_transform(labels["class"])
    return class_labels
