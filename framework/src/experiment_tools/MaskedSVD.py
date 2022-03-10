import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

class MaskedSVD(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, mask=None):
        # mask should contain selected cols. Suppose it is boolean to avoid code overhead
        self.n_components = n_components
        self.mask = mask

    def fit(self, X, y = None):
        self.svd = TruncatedSVD(n_components=self.n_components)
        mask = self.mask
        mask = self.mask if self.mask is not None else slice(None)
        self.svd.fit(X[:, mask])
        return self

    def transform(self, X, y = None):
        mask = self.mask if self.mask is not None else slice(None)
        svd_transformed = self.svd.transform(X[:, mask])
        if self.mask is not None:
            remaining_cols = X[:, ~mask]
            return np.hstack([remaining_cols, svd_transformed])
        else:
            return svd_transformed