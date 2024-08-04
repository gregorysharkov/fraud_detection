import re
from functools import cached_property
from typing import List, Union

import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from gensim.models import FastText



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


def _process_column(column: pd.Series) -> pd.Series:
    """
    wrapper function for a single column: cleans values, splits and pads them

    Args:
        series: a pandas series containing column values

    Returns:
        processed series
    """
    sentences = column.apply(_clean_column_value)
    max_sentence_length = max(len(x) for x in sentences)
    sentences = sentences.apply(lambda x: _pad_sentence(x, max_sentence_length=max_sentence_length))
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
    return np.concatenate(embedded_sentences)


class TextEmbedder:
    """class responsible for encoding textual data using fast text"""
    def __init__(self, sentences: pd.Series, model_settings: dict) -> None:
        """
        Initializes a TextEmbedder object with sentences and model settings.

        Parameters:
        - sentences (pd.Series): A pandas Series containing textual data to be embedded.
        - model_settings (dict): A dictionary containing settings for the FastText model.

        Returns:
        - None
        """
        self.sentences = sentences
        self.model_settings = model_settings

    @property
    def vector_size(self) -> int:
        return self.model_settings.get("vector_size", 3)

    @cached_property
    def processed_sentences(self):
        return _process_column(self.sentences)

    @property
    def max_sentence_length(self):
        return len(self.processed_sentences[0])
    
    def fit_embedder(self) -> None:
        """
        Fits and trains a FastText model using the processed sentences and model settings.
        Also initializes an embedding for an empty string in the FastText model's vocabulary.

        Parameters:
        - self: The instance of the TextEmbedder class.

        Returns:
        - None
        """
        self.model = _train_fast_text_model(self.processed_sentences, **self.model_settings)
        self.model.wv[""] = np.array([0.] * self.vector_size)

    def transform_sentence(self, sentence: str) -> list[float]:
        """
        Transforms a single sentence into a list of float values using the trained FastText model.

        Parameters:
        - sentence (str): The sentence to be transformed. It should be a string of words separated by spaces.

        Returns:
        - list[float]: A list of float values representing the embedded sentence. The length of the list is determined by the
        maximum sentence length and the vector size of the FastText model.
        """
        return _process_single_sentence(sentence, self.max_sentence_length, self.model)


class FastTextEncoder(BaseEstimator, TransformerMixin):
    """
    Class responsible for encoding a single categorical column using TextEmbedder.
    """

    def __init__(self, column: str, embedder_settings: dict) -> None:
        """
        Initializes a FastTextEncoder object with the column name and embedder settings.

        Parameters:
        - column (str): The name of the column to be encoded.
        - embedder_settings (dict): The settings for the TextEmbedder.

        Returns:
        - None
        """
        self.column = column
        self.embedder_settings = embedder_settings
        self.embedder = None

    def fit(self, X, y=None):
        """
        Fits the TextEmbedder using the sentences from the specified column.

        Parameters:
        - X (DataFrame): The input data containing the column to be encoded.
        - y (None): Not used in this method.

        Returns:
        - self: The fitted FastTextEncoder object.
        """
        sentences = X[self.column] # .apply(lambda x: x.split()) # .tolist()
        self.embedder = TextEmbedder(sentences, self.embedder_settings)
        self.embedder.fit_embedder()
        return self

    def transform(self, X):
        """
        Transforms the specified column into a list of float values using the fitted TextEmbedder.

        Parameters:
        - X (DataFrame): The input data containing the column to be transformed.

        Returns:
        - list[float]: The transformed column as a list of float values.
        """

        if isinstance(X, pd.DataFrame):
            return X[self.column].apply(lambda x: self.embedder.transform_sentence(x))

        return self.embedder.transform_sentence(X)


def get_row_preprocessor(cat_columns: list[str], embedder_settings: dict) -> Pipeline:
    """
    generates a pipeline to preprocess the incomming data
    
    Args:
        id_columns: list of columns to be treated as id columns
        cat_columns: list of columns to be treated as categorical columns
        embedding_size: size of embedding vector for each column
        window_size: size of the window for fraud detection
    
    Returns:
        a pipeline that transforms a single row into an array after embedding
    """

    preprocessors = {f"emb_{x}": FastTextEncoder(x, embedder_settings) for x in cat_columns}
    feature_union = FeatureUnion(
        transformer_list = [(key, preprocessor) for key, preprocessor in preprocessors.items()]
    )


    return Pipeline(
        steps=[
            ("combined_embeddings", feature_union),
            # ("row_selector", row_selector),
        ],
    )


# class CombinedRowTransformer(BaseEstimator, TransformerMixin):
#     """class responsible for preprocessing a single row"""
#     def __init__(self, n_rows: int, id_columns: list[str], columns: list[str]) -> None:
#         self.n_rows = n_rows
#         self.columns = columns
#         self.id_columns = id_columns
    
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X):
#         grouped_data = X.groupby(self.id_columns)
#         transformed_data = grouped_data.apply(lambda group: self._combine_rows(group, self.n_rows, self.columns))
#         transformed_data = pd.concat(transformed_data.values)

#         return transformed_data
    
#     def _combine_rows(self, group, n_rows, columns):
#         bottom_n_rows = group.tail(n_rows)
#         if len(bottom_n_rows) < n_rows:
#             padding = pd.DataFrame(0, index=np.arange(n_rows - len(bottom_n_rows)), columns=columns)
#             bottom_n_rows = pd.concat([padding, bottom_n_rows])

#         matrix = bottom_n_rows[columns].values
#         return matrix




    # row_selector = CombinedRowTransformer(
    #     n_rows=window_size,
    #     id_columns=id_columns,
    #     columns=cat_columns,
    # )