from sklearn.base import BaseEstimator, TransformerMixin

from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import RootMeanSquaredError, Accuracy
from tensorflow.keras.optimizers import SGD, Adam
from extra_keras_metrics import get_standard_binary_metrics
from tensorflow.keras import regularizers
class KerasSklearnWrapper(BaseEstimator, TransformerMixin):

    def baseline_model(self):
        # create model
        model = Sequential()

        model.add(Dense(8, activation='relu', kernel_initializer='ones'))
        model.add(Dense(1, activation='sigmoid'))

        # Compile model
        model.compile(loss="binary_crossentropy", optimizer='nadam', metrics=get_standard_binary_metrics())
        return model
