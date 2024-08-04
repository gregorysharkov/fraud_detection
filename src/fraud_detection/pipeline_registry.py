"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from fraud_detection.pipelines.feature_engineering import create_pipeline as create_features_pipeline


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """

    feature_engineering_pipeline = create_features_pipeline()

    return {
        "create_features": feature_engineering_pipeline,
        "__default__": feature_engineering_pipeline,
    }