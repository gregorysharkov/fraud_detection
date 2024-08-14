from fraud_detection.utils.text_embedder import TextEmbedder


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
        sentences = X[self.column]
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
            return X[self.column].apply(lambda x: self.embedder.transform_sentence(x)).apply(pd.Series)

        return pd.Series(self.embedder.transform_sentence(X))