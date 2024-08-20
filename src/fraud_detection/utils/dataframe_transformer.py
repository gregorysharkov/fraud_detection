from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, id_columns: list[str]) -> None:
        self.id_columns = id_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print(f"{pd.DataFrame(X)=}")
        transformed_data = (
            pd.DataFrame(X)
            .apply(lambda row: _transform_row(row, self.id_columns), axis=1)
            .apply(pd.Series)
        )
        transformed_data.columns = ["features"] + self.id_columns
        return transformed_data


def _transform_row(row, id_cols: list[str]) -> tuple[list, Any, Any]:
    """Transforms a single row into a list"""
    row = row.values.tolist()
    features = row[:-len(id_cols)]
    id_values = row[-len(id_cols):]
    return (features, *id_values)
