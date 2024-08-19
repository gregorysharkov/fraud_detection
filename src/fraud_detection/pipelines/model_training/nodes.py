import pandas as pd
from sklearn.pipeline import Pipeline

def train_model(train_data: pd.DataFrame, train_params: dict) -> Pipeline:
    """
    function trains a model with the given set of parameters

    Args:
        train_data: dataset used to train the model
        train_params: list of parameters used for model training
    """