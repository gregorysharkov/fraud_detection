"""Project pipelines."""

from typing import Dict

from kedro.pipeline import Pipeline

from fraud_detection.pipelines.data_preprocessing.pipeline import (
    create_pipeline as data_processing_pipeline,
)
from fraud_detection.pipelines.model_training.pipeline import (
    create_pipeline as model_training_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    preprocessing = data_processing_pipeline()
    model_training = model_training_pipeline()
    return {
        "feature_engineering": preprocessing,
        "model_training": model_training,
        "__default__": preprocessing + model_training,
    }
