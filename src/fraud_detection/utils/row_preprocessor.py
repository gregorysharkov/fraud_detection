from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from fraud_detection.utils.fast_text_encoder import FastTextEncoder
from fraud_detection.utils.concatenate_transformer import ConcatenateTransformer
from fraud_detection.utils.dataframe_transformer import DataFrameTransformer
from fraud_detection.utils.row_combination_utils import combine_features, select_single_label


# TODO: move the fast model settings to the configuration file
FAST_MODEL_SETTINGS = {
    "vector_size": 3,
    "window": 5,
    "min_count": 3,
    "workers": 4,
}

ID_COLS = ["cc_num", "window_id", "label",]


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

    preprocessors = _generate_encoder_pipeline(cat_columns, embedder_settings)
    numeric_pipelines = _generate_numeric_pipeline(num_columns)
    preprocessors.extend(numeric_pipelines)
    id_pipelines = _generate_id_columns_pipeline(ID_COLS)
    preprocessors.extend(id_pipelines)

    feature_union = FeatureUnion(transformer_list=preprocessors, n_jobs=1, verbose=True, verbose_feature_names_out=True)

    return Pipeline(
        steps=[
            ("combined_embeddings", feature_union),
            ("concatenate_features", ConcatenateTransformer()),
            ("convert_to_dataframe", DataFrameTransformer(ID_COLS)),
        ],
    )


def _generate_encoder_pipeline(cat_columns: list[str], embedder_settings: dict) -> list[tuple[str, Pipeline]]:
    """
    Generates a single encoder pipeline for a specific column.

    Args:
        cat_columns: List of columns to be treated as categorical columns.
        embedder_settings: Settings for the TextEmbedder.

    Returns:
        A tuple containing the column name and the encoder pipeline.
    """
    return [
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


def _generate_numeric_pipeline(num_columns: list[str]) -> list[tuple[str, Pipeline]]:
    """
    Generates a single numeric pipeline for a specific column.

    Args:
        num_columns: List of columns to be treated as numerical columns.

    Returns:
        A tuple containing the column name and the numeric pipeline.
    """
    return [
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


def _generate_id_columns_pipeline(id_cols: list[str]) -> list[tuple[str, Pipeline]]:
    """
    Generates a single pipeline for ID columns.

    Args:
        id_cols: List of columns to be treated as ID columns.

    Returns:
        A tuple containing the column name and the ID pipeline.
    """
    return [
        (
            f"id_pipeline_{x}",
            Pipeline(
                [
                    (f"selector_{x}", FunctionTransformer(lambda X: X[id_cols], validate=False)),
                ]
            )
        )
        for x in id_cols
    ]
