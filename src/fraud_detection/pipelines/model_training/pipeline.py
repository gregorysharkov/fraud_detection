"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_torch_device, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_torch_device, inputs=None, outputs="device", name="get_device"
            ),
            node(
                func=train_model,
                inputs=["raw_data", "params:train_params", "device"],
                outputs="trained_model",
                name="train_model",
            ),
        ]
    )
