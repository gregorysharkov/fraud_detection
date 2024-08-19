from typing import Any, Iterable
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin


class ConcatenateTransformer(BaseEstimator, TransformerMixin):
    """this transformer is responsible to wrap the flatten nested list function"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return_list = []
        for x in X:
            return_list.append(flatten_nested_list(x))

        return return_list


def flatten_nested_list(nested_list: Iterable) -> list[Any]:
    """
    Function flatterns a nested list
    """
    flattened_list = []
    for element in nested_list:
        element_items = []
        if isinstance(element, np.ndarray):
            sub_element_items = flatten_nested_list(element)
            element_items.extend(sub_element_items)
        else:
            element_items.append(element)

        flattened_list.extend(element_items)
        
    return flattened_list

