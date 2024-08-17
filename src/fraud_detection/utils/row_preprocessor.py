from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraud_detection.utils.fast_text_encoder import FastTextEncoder
from fraud_detection.utils.concatenate_transformer import ConcatenateTransformer


# TODO: move the fast model settings to the configuration file
FAST_MODEL_SETTINGS = {
    "vector_size": 3,
    "window": 5,
    "min_count": 3,
    "workers": 4,
}


def get_row_preprocessor(
    cat_columns: list[str],
    num_columns: list[str],
    embedder_settings: dict,
) -> Pipeline:
    """
    Generates a pipeline to preprocess the incoming data.

    Args:
        cat_columns: List of columns to be treated as categorical columns.
        num_columns: List of columns to be treated as numerical columns.
        embedder_settings: Settings for the TextEmbedder.

    Returns:
        A pipeline that transforms a single row into an array after embedding.
    """

    preprocessors = [
        (
            f"encoder_pipeline_{x}",
            Pipeline(
                [
                    (f"encoder_{x}", FastTextEncoder(x, embedder_settings)),
                ]
            )
        )
        for x in cat_columns
    ]

    preprocessors.extend(
        [
            (
                f"numeric_pipeline_{x}",
                Pipeline(
                    [
                        (f"selector_{x}", FunctionTransformer(lambda X: X[[x]], validate=False)),
                        (f"scaler_{x}", StandardScaler()),
                        (f"imputer_{x}", SimpleImputer(strategy="mean")),
                    ]
                )
            )
            for x in num_columns
        ]
    )
    feature_union = FeatureUnion(transformer_list=preprocessors, n_jobs=1, verbose=True, verbose_feature_names_out=True)

    return Pipeline(
        steps=[
            ("combined_embeddings", feature_union),
            ("concatenate_features", ConcatenateTransformer()),
        ],
    )