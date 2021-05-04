import numpy as np
from tqdm import tqdm
from jordiyapz_ai_toolkit.method import Validation


class Activation:

    class Sigmoid:

        @staticmethod
        def name():
            return 'sigmoid'

        @staticmethod
        def c(Z):
            return 1 / (1 + np.exp(-Z))

        @staticmethod
        def d(Z):
            S = Activation.Sigmoid.c(Z)
            return S * (1 - S)

    class Relu:

        @staticmethod
        def name():
            return 'relu'

        @staticmethod
        def c(Z):
            return np.maximum(Z, 1e-5)

        @staticmethod
        def d(Z):
            return np.where(Z < 0, 0, 1)

    class Softmax:

        @staticmethod
        def name():
            return 'softmax'

        @staticmethod
        def c(Z):
            ez = np.exp(Z)
            return ez / np.sum(ez, axis=1, keepdims=True)


class Layer:

    class Input:

        def __init__(self, n):
            self.n = n

        def ignite(self, n):
            pass

        def set_learning_rate(self, learning_rate):
            pass

        def activate(self, X):
            return X

        def propagate(self, *args):
            return None

        def optimize(self):
            pass

    class Linear:

        def __init__(self, n, activation='sigmoid'):
            self.n = n
            self.learning_rate = .001
            self.weight = None
            if activation is 'relu':
                self.activation = Activation.Relu
            elif activation is 'sigmoid':
                self.activation = Activation.Sigmoid
            else:
                raise Exception('Tolong pilih fungsi aktivasi yang benar')

        def ignite(self, inp_m):
            self.m = inp_m
            self.weight = np.random.randn(self.n, inp_m)
            self.b = np.zeros((self.n, 1))

        def set_learning_rate(self, learning_rate):
            self.learning_rate = learning_rate

        def activate(self, A_prev):
            self.A_prev = A_prev
            self.Z = np.dot(self.weight, A_prev) + self.b
            return self.activation.c(self.Z)

        def propagate(self, dA):
            try:
                dZ = dA * self.activation.d(self.Z)
                self.dW = np.dot(dZ, self.A_prev.T) / self.m
                self.db = np.sum(dZ, axis=1, keepdims=True) / self.m
                dA_prev = np.dot(self.weight.T, dZ)
                return dA_prev
            except:
                print('eexceptioon')

        def optimize(self):
            self.weight = self.weight - self.learning_rate * self.dW
            self.b = self.b - self.learning_rate * self.db

# lin  = Layer.Linear(5, activation='relu')
# lin.ignite(6)
# lin.activate(np.array([[0, 3, 5, 6, 3, 7]]).T)


class NyuNet:  # nyural netwok

    def __init__(self, layers):
        self.layers = layers
        self.__ignited = False
        if isinstance(self.layers[0], Layer.Input):
            self.__ignite(self.layers[0].n)

    def __ignite(self, n):
        np.random.seed(1)
        for layer in self.layers:
            layer.ignite(n)
            n = layer.n
        self.__ignited = True

    def feedforward(self, X):
        assert self.__ignited, 'Belum terinisialisasi. Gimana sih!?'
        A = X
        for layer in self.layers:
            A = layer.activate(A)
        return A

    def __bp(self, y_pred, y):
        dA = (1-y)/(1-y_pred) - y/y_pred
        for layer in reversed(self.layers):
            dA = layer.propagate(dA)
            layer.optimize()

    def __loss(self, y_pred, y):
        loss = -(np.dot(y, np.log(y_pred).T) +
                 np.dot(1-y, np.log(1-y_pred).T)) / y.shape[1]
        return loss

    def fit(self, X, y, epochs=5, learning_rate=None):
        if not self.__ignited:
            self.__ignite(X.shape[1])

        if learning_rate:
            for layer in self.layers:
                layer.set_learning_rate(learning_rate)

        X = X.T
        y = y.T
        hist = []
        for _ in tqdm(range(epochs), 'Epochs'):
            y_pred = self.feedforward(X)
            loss = self.__loss(y_pred, y)
            hist.append((loss / loss.shape[0]**2).sum())
            self.__bp(y_pred, y)
        return hist

    def predict(self, x):
        assert self.__ignited, 'Model belum terinisialisasi'
        return np.round(self.feedforward(x.reshape(x.shape[0], 1)))

    def __get_metric(self, name, y_pred, y_test):
        if name is 'loss':
            return self.__loss(y_pred, y_test)
        if name is 'accuracy':
            return Validation.accuracy(y_test.flatten(), np.round(y_pred).flatten())

    def evaluate(self, X_test, y_test, metrics=['accuracy']):
        y_test = y_test.T
        y_pred = self.feedforward(X_test.T)
        return tuple(self.__get_metric(name, y_pred, y_test) for name in metrics)
