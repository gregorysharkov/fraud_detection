import re
from typing import List, Union

import pandas as pd

from gensim.models import FastText



def clean_column_value(value: Union[str, pd.Series]) -> Union[str, List[str]]:
    """cleans column value from noise"""
    if isinstance(value, pd.Series):
        return value.apply(clean_column_value)
    return_value = value.lower()
    return_value = re.sub("^fraud_", "", return_value)
    return_value = re.sub(r"[^a-z0-9]+?", " ", return_value)
    return_value = re.sub(" +", " ", return_value)
    return_value = return_value.split()
    return return_value


def pad_sentence(sentence: list[str], max_sentence_length: int) -> list[str]:
    """
    function pads a sentence with empty strings
    the sentence should be split into words. The padding will be done with empty strings

    Args:
        sentence: list of strings
        max_sentence_length: maximum length of the sentence

    Returns:
        list of strings padded with empty strings
    """
    if len(sentence) > max_sentence_length:
        return sentence[:max_sentence_length]

    return sentence + [''] * (max_sentence_length - len(sentence))


def train_fast_text_model(sentences: list[str], **fast_text_kwargs) -> FastText:
    """trains a fast text model given settings"""

    return FastText(sentences, **fast_text_kwargs)


def embed_sentence(sentence: str, fast_model: FastText) -> list[list[float]]:
    """
    function embeds a single sentence using the fast text model
    
    Args:
        sentence: string sentence

    Returns combined embeddings of a sentence 2d list
    """
    embeddings = [fast_model.wv[x] for x in sentence]
    return embeddings


def process_list_of_sentences(sentences: pd.Series) -> pd.Series:
    """
    Processes a list of sentences by cleaning and padding them.

    Parameters:
    - sentences (pd.Series): A pandas Series containing textual data to be processed.

    Returns:
    - pd.Series: A pandas Series containing processed sentences.
    """
    processed_sentences = sentences.apply(clean_column_value)
    max_sentence_length = max(processed_sentences.apply(len))
    processed_sentences = processed_sentences.apply(lambda x: pad_sentence(x, max_sentence_length))
    return processed_sentences


def process_single_sentence(sentence: str, max_sentence_length: int, fast_model: FastText) -> list[str]:
    """
    Another wrapper for preprocessing functions, but this time functions get applied to a single sentence

    Args:
        sentence: string containg words to be embedded
        max_sentence_length: maximum number of words to be kept in a single sentence

    Returns:
        preprocessed sentence
    """

    return_sentence = clean_column_value(sentence)
    return_sentence = pad_sentence(return_sentence, max_sentence_length)
    embedded_sentences = embed_sentence(return_sentence, fast_model)
    return [' '.join(map(str, x)) for x in embedded_sentences]

