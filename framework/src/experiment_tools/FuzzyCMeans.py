import skfuzzy as fuzz
from sklearn.base import BaseEstimator, TransformerMixin


class FuzzyCMeans(BaseEstimator, TransformerMixin):
    def __init__(self, n_centers=2, m=2, error=0.0005, maxiter=1000):
        self.n_centers = n_centers
        self.error = error
        self.maxiter = maxiter
        self.m = m

    def fit(self, X, y=None):
        self.cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, self.n_centers, self.m, error=self.error,
                                                              maxiter=self.maxiter, init=None)
        return self

    def predict(self, X):
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(X.T, self.cntr, self.m, error=self.error,
                                                           maxiter=self.maxiter)
        cluster_membership = np.argmax(u, axis=0) + 1
        return cluster_membership