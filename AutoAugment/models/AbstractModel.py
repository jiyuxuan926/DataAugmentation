import numpy as np
from abc import ABC, abstractmethod

def create_batches(X, Y, batch_size=32):
    n_samples = X.shape[0]
    n_batches = n_samples // batch_size

    batches = [(X[i * batch_size: (i + 1) * batch_size, :, :, :],
                Y[i * batch_size: (i + 1) * batch_size]) for i in range(n_batches)]

    return batches

def random_batch(X, Y, batch_size = 32):
    num_train = X.shape[0]
    idx = np.random.randint(low = 0, high = num_train, size = batch_size)

    return X[idx], Y[idx]

class AbstractModel(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def train(self, subpolicies, X, Y):
        pass

    @abstractmethod
    def test(self, X, Y):
        pass