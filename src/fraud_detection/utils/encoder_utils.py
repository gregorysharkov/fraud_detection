import re
from typing import List, Union

import pandas as pd

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from gensim.models import FastText

from fraud_detection.utils.fast_text_encoder import FastTextEncoder


def _clean_column_value(value: Union[str, pd.Series]) -> Union[str, List[str]]:
    """cleans column value from noise"""
    if isinstance(value, pd.Series):
        return value.apply(_clean_column_value)
    return_value = value.lower()
    return_value = re.sub("^fraud_", "", return_value)
    return_value = re.sub(r"[^a-z0-9]+?", " ", return_value)
    return_value = re.sub(" +", " ", return_value)
    return_value = return_value.split()
    return return_value


def _pad_sentence(sentence: list[str], max_sentence_length: int) -> list[str]:
    """function pads a sentence with empty strings"""
    if len(sentence) > max_sentence_length:
        return sentence[:max_sentence_length]

    return sentence + [''] * (max_sentence_length - len(sentence))


FAST_MODEL_SETTINGS = {
    "vector_size": 3,
    "window": 5,
    "min_count": 3,
    "workers": 4,
}


def _train_fast_text_model(sentences: list[str], **fast_text_kwargs) -> FastText:
    """trains a fast text model given settings"""

    return FastText(sentences, **fast_text_kwargs)


def _embed_sentence(sentence: str, fast_model: FastText) -> list[list[float]]:
    """
    function embeds a single sentence using the fast text model
    
    Args:
        sentence: string sentence

    Returns combined embeddings of a sentence 2d list
    """
    embeddings = [fast_model.wv[x] for x in sentence]
    return embeddings


def _process_list_of_sentences(column: Union[pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Wrapper function for a single column: cleans values, splits and pads them

    Args:
        column: a pandas Series or DataFrame containing column values

    Returns:
        processed DataFrame
    """
    if isinstance(column, pd.Series):
        column = pd.DataFrame(column)

    sentences = column.apply(_clean_column_value)
    max_sentence_length = max(len(x) for x in sentences)
    sentences = sentences.apply(
        lambda x: _pad_sentence(x, max_sentence_length=max_sentence_length)
    )
    return sentences


def _process_single_sentence(sentence: str, max_sentence_length: int, fast_model: FastText) -> list[str]:
    """
    Another wrapper for preprocessing functions, but this time functions get applied to a single sentence

    Args:
        sentence: string containg words to be embedded
        max_sentence_length: maximum number of words to be kept in a single sentence

    Returns:
        preprocessed sentence
    """

    return_sentence = _clean_column_value(sentence)
    return_sentence = _pad_sentence(return_sentence, max_sentence_length)
    embedded_sentences = _embed_sentence(return_sentence, fast_model)
    return [' '.join(map(str, x)) for x in embedded_sentences]


def get_row_preprocessor(
    cat_columns: list[str],
    num_columns: list[str],
    embedder_settings: dict,
) -> Pipeline:
    """
    generates a pipeline to preprocess the incoming data
    
    Args:
        id_columns: list of columns to be treated as id columns
        cat_columns: list of columns to be treated as categorical columns
        num_columns: list of columns to be treated as numerical columns
        embedder_settings: settings for the TextEmbedder
    
    Returns:
        a pipeline that transforms a single row into an array after embedding
    """

    preprocessors = [
        (
            f"encoder_pipeline_{x}",
            Pipeline(
                [
                    (f"encoder_{x}", FastTextEncoder(x, embedder_settings))
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
                        (f"selector_{x}", FunctionTransformer(lambda X: X[x], validate=False)),
                        (f"scaler_{x}", StandardScaler()),
                        (f"imputer_{x}", SimpleImputer(strategy="mean")),

                    ]
                )
            )
            for x in num_columns
        ]
    )
    feature_union = FeatureUnion(transformer_list=preprocessors, n_jobs=1)

    return Pipeline(
        steps=[
            ("combined_embeddings", feature_union),
        ],
    )