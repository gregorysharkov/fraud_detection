from fraud_detection.utils.encoder_utils import _process_list_of_sentences, _process_single_sentence, _train_fast_text_model


import numpy as np
import pandas as pd


from functools import cached_property


class TextEmbedder:
    """class responsible for encoding textual data using fast text"""
    def __init__(self, sentences: pd.DataFrame, model_settings: dict) -> None:
        """
        Initializes a TextEmbedder object with sentences and model settings.

        Parameters:
        - sentences (pd.DataFrame): A pandas DataFrame containing textual data to be embedded.
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
        return _process_list_of_sentences(self.sentences).apply(lambda x: ' '.join(x)).tolist()

    @property
    def max_sentence_length(self):
        return len(self.processed_sentences[0].split())

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