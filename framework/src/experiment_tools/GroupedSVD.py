import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD

class GroupedSVD(BaseEstimator, TransformerMixin):

    def __init__(self, n_components=2, mask=None):
        # mask should contain selected cols. Suppose it is boolean to avoid code overhead
        self.n_components = n_components
        self.mask = mask

        n_groups = np.max(self.mask)
        self.indices = []
        for i in range(0, n_groups+1):
            indeces_list = []
            for j, k in enumerate(self.mask):
                if k == i:
                    indeces_list.append(j)
            self.indices.append(indeces_list)

    def fit(self, X, y = None):
        self.svd_list = []
        for i, idx in enumerate(self.indices[1:]):
                svd = TruncatedSVD(n_components=self.n_components)
                svd.fit(X[:, idx])
                self.svd_list.append(svd)
        return self

    def transform(self, X, y = None):
        transformed_cols = X[:, self.indices[0]]
        for k, idx in enumerate(self.indices[1:]):
            svd_transformed = self.svd_list[k].transform(X[:, idx])
            transformed_cols = np.hstack([transformed_cols, svd_transformed])
        return transformed_cols