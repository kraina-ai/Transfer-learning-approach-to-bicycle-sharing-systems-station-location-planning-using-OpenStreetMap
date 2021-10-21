import os

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from tensorflow.keras import callbacks, losses
from tensorflow.keras.models import Model

tf.get_logger().setLevel('WARNING')
tf.autograph.set_verbosity(1)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#      tf.config.experimental.set_memory_growth(gpu, True)

class Autoencoder(Model):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(input_size, activation='relu'),
            tf.keras.layers.Dense(20, activation='relu'),
            tf.keras.layers.Dense(2)
        ])

    def call(self, x):
        return self.model(x)

class TensorflowEstimatorN(BaseEstimator):
    def __init__(self):
        self.model = Autoencoder(self.input_size)
        self.enc = OneHotEncoder(handle_unknown='ignore')

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True

        _y = y.reshape(-1,1)

        y_one_hot = self.enc.fit_transform(_y).toarray()
        # print(X.shape, y_one_hot.shape)
        # print(y_one_hot)

        # `fit` should always return `self`
        callback = callbacks.EarlyStopping(monitor='loss', patience=10)
        self.model.compile(optimizer='adam', loss=losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])
        self.model.fit(X, y_one_hot,
                    epochs=200,
                    callbacks=[callback],
                    shuffle=True
                    ,verbose=0)
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        y_pred = self.model.call(X)
        results = np.array([np.argmax(y) for y in y_pred])
        # print(y_pred.shape, results.shape)
        return results

class TensorflowEstimator20(TensorflowEstimatorN):
    input_size = 20

class TensorflowEstimator32(TensorflowEstimatorN):
    input_size = 32

class TensorflowEstimator36(TensorflowEstimatorN):
    input_size = 36

class TensorflowEstimator50(TensorflowEstimatorN):
    input_size = 50

class TensorflowEstimator64(TensorflowEstimatorN):
    input_size = 64

class TensorflowEstimator100(TensorflowEstimatorN):
    input_size = 100

class TensorflowEstimator128(TensorflowEstimatorN):
    input_size = 128

class TensorflowEstimator200(TensorflowEstimatorN):
    input_size = 200

class TensorflowEstimator256(TensorflowEstimatorN):
    input_size = 256

class TensorflowEstimator300(TensorflowEstimatorN):
    input_size = 300

class TensorflowEstimator500(TensorflowEstimatorN):
    input_size = 500
