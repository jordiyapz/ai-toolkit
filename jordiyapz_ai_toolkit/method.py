import numpy as np
import pandas as pd


class Distance:

    @staticmethod
    def manhattan(X, x_pred):
        return abs(X - x_pred).sum(axis=1)

    @staticmethod
    def minkowski(h=1.5):
        return lambda X, x_pred: ((X - x_pred) ** h).sum(axis=1) ** (1/h)

    @staticmethod
    def euclidean(X, x_pred):
        return Distance.minkowski(2)(X, x_pred)


class Validation:

    @staticmethod
    def sse(y, h):
        delta = np.array(h - y)
        delta = np.reshape(delta, (delta.shape[0], -1))
        return np.dot(delta.T, delta).flatten()[0]

    @staticmethod
    def accuracy(y, h):
        delta = abs(h - y).sum() / y.shape
        return (1 - delta)

    @staticmethod
    def confusion_matrix(y_true, y_pred):
        return pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
