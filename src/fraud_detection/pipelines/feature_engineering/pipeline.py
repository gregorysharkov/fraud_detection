"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.6
"""

from kedro.pipeline import Pipeline, pipeline, node
from fraud_detection.pipelines.feature_engineering.nodes import prepare_train_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=prepare_train_data,
                inputs = {
                    "raw_data": "train_data",
                    "train_set_params": "params:train_set_parameters",
                },
                outputs="combined_train_data",
                name="generate_training_data",
            )
        ]
    )
