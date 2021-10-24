import numpy as np
from sklearn.preprocessing import MinMaxScaler


def normalize_data(x: np.ndarray) -> np.ndarray:
    new_array = np.zeros(x.shape)
    # iterate over columns and normalize over whole provided dataset
    for i in range(x.shape[1]):
        vector = x[:,i].reshape((-1,1))
        scaler = MinMaxScaler()
        new_array[:,i] = scaler.fit_transform(vector).reshape((-1,))
    return new_array
