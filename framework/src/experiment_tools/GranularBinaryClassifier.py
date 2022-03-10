import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier


class GranularBinaryClassifier (BaseEstimator, TransformerMixin):
    def __init__(self, estimator=RandomForestClassifier(), binary_threshold=2):
        self.binary_threshold = binary_threshold
        self.estimator = estimator

    def recursive_ensamble(self, X, y, binary_threshold, dict_estimators, key="0"):
        if len(np.unique(y)) == 1:
            dict_estimators[key] = int(np.unique(y)[0])
            return dict_estimators
        else:
            class_0_idx = np.where(y <= binary_threshold)
            class_1_idx = np.where(y > binary_threshold)

            data_0 = X[class_0_idx, :]
            data_0 = data_0.reshape((data_0.shape[1], data_0.shape[2]))
            labels_0 = y[class_0_idx]

            data_1 = X[class_1_idx, :]
            data_1 = data_1.reshape((data_1.shape[1], data_1.shape[2]))
            labels_1 = y[class_1_idx]

            sub_x = np.vstack((data_0, data_1))
            sub_y = [0 for r in range(0, data_0.shape[0])] + [1 for r in range(0, data_1.shape[0])]

            t_0 = (np.min(labels_0) + np.max(labels_0)) // 2
            t_1 = (np.min(labels_1) + np.max(labels_1)) // 2

            dict_estimators[key] = self.estimator
            dict_estimators[key].fit(sub_x, sub_y)

            dict_estimators = self.recursive_ensamble(data_0, labels_0, t_0, dict_estimators, key + "0")
            dict_estimators = self.recursive_ensamble(data_1, labels_1, t_1, dict_estimators, key + "1")
        return dict_estimators

    def fit(self, X, y=None):
        self.dict_estimators = self.recursive_ensamble(X, y, self.binary_threshold, {})

        return self

    def predict(self, X):
        predictions = []
        for index in range(0, X.shape[0]):
            key = "0"
            while type(self.dict_estimators[key]) != int:
                p = self.dict_estimators[key].predict([X[index, :]])[0]
                key = key + str(p)

            predictions.append(self.dict_estimators[key])
        return np.array(predictions)