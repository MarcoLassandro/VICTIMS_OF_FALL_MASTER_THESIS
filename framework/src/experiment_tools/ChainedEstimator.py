import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor


class ChainedEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, regressor=GradientBoostingRegressor(), clf=RandomForestClassifier(), binary_threshold=2):
        self.binary_threshold = binary_threshold
        self.regressor = regressor
        self.clf = clf

    def fit(self, X, y=None):
        self.regressor.fit(X, y)

        self.sub_estimators = {}
        for i in range(2, 7, 1):
            class_0_idx = np.where(y == i - 1)
            class_1_idx = np.where(y == i)
            class_2_idx = np.where(y == i + 1)

            data_0 = X[class_0_idx, :]
            data_0 = data_0.reshape((data_0.shape[1], data_0.shape[2]))

            data_1 = X[class_1_idx, :]
            data_1 = data_1.reshape((data_1.shape[1], data_1.shape[2]))

            data_2 = X[class_2_idx, :]
            data_2 = data_2.reshape((data_2.shape[1], data_2.shape[2]))

            sub_x_1 = np.vstack((data_0, data_1))
            sub_y_1 = [0 for r in range(0, data_0.shape[0])] + [1 for r in range(0, data_1.shape[0])]
            clf_1 = self.clf
            clf_1.fit(sub_x_1, sub_y_1)
            self.sub_estimators[str(i - 1) + str(i)] = clf_1

            sub_x_2 = np.vstack((data_1, data_2))
            sub_y_2 = [1 for r in range(0, data_1.shape[0])] + [0 for r in range(0, data_2.shape[0])]
            clf_2 = self.clf
            clf_2.fit(sub_x_2, sub_y_2)
            self.sub_estimators[str(i) + str(i + 1)] = clf_2

            sub_x_3 = np.vstack((data_0, data_2))
            sub_y_3 = [0 for r in range(0, data_0.shape[0])] + [1 for r in range(0, data_2.shape[0])]
            clf_3 = self.clf
            clf_3.fit(sub_x_3, sub_y_3)
            self.sub_estimators[str(i - 1) + str(i + 1)] = clf_3

        return self

    def predict(self, X):
        regressor_preds = self.regressor.predict(X)
        clf_predictions = []
        for index, p in enumerate(regressor_preds):
            p = int(np.round(p))
            if p > 7:
                p = 7
            clf_preds = [self.sub_estimators[k].predict([X[index, :]]) for k in self.sub_estimators.keys() if
                         str(p) in k]

            if p == 1:
                p = p if clf_preds[0] == 0 else p + 1
            elif p == 7:
                p = p if clf_preds[0] == 0 else p - 1
            elif len(clf_preds) > 1:
                if clf_preds[0] == 0 or clf_preds[1] == 0:
                    p = p - 1 if self.sub_estimators[str(p - 1) + str(p + 1)].predict([X[index, :]]) == 0 else p + 1
            clf_predictions.append(p)
        return np.array(clf_predictions)