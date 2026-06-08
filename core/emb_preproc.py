import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler


class RowStandardScaler(TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        assert isinstance(X, np.ndarray)
        return StandardScaler().fit_transform(X.T).T
