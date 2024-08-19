import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameToListTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X).apply(lambda row: row.tolist(), axis=1)
