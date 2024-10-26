"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import download_data, train_test_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=download_data,
                inputs=None,
                outputs="raw_data",
                name="download_data",
            ),
            # node(
            #     func=train_test_split,
            #     inputs=["raw_data"],
            #     outputs=["train_data", "test_data"],
            #     name="train_test_split",
            # ),
        ]
    )
