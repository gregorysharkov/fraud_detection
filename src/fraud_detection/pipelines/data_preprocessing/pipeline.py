"""
This is a boilerplate pipeline 'data_preprocessing'
generated using Kedro 0.19.9
"""

from kedro.pipeline import Pipeline, node, pipeline

from fraud_detection.pipelines.data_preprocessing import nodes as nd


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=nd.generate_client_mapper,
                inputs="raw_labels",
                outputs="client_mapper",
                name="generate_client_mapper",
            ),
            node(
                func=nd.process_edges,
                inputs=["raw_edges", "client_mapper"],
                outputs="preprocessed_edges",
                name="process_edges",
            ),
            node(
                func=nd.preprocess_labels,
                inputs=["raw_labels"],
                outputs="preprocessed_labels",
                name="preprocess_labels",
            ),
            node(
                func=nd.preprocess_features,
                inputs=["raw_features"],
                outputs="preprocessed_features",
                name="preprocess_features",
            ),
            node(
                func=nd.collect_training_data,
                inputs=[
                    "preprocessed_labels",
                    "preprocessed_edges",
                    "preprocessed_features",
                    "params:train_ratio",
                    "params:val_ratio",
                ],
                outputs="processed_data",
                name="collect_training_data",
            ),
        ]
    )
