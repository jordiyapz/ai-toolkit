import numpy as np
from jordiyapz_ai_toolkit.method import Distance, Validation


class Knn:

    @staticmethod
    def get_distances(X, y, X_test, distance_func=Distance.manhattan):
        # unsorted distances
        return np.array([distance_func(X, X_test.iloc[i]) for i in range(X_test.shape[0])])

    @staticmethod
    def predict(X, y, X_test, K=3, distance_func=Distance.manhattan):
        D = Knn.get_distances(X, y, X_test, distance_func).argsort()
        return np.array([y.iloc[d][:K].mode()[0] for d in D])

    @staticmethod
    def validate(X, y, X_test, y_truth, K=3, distance_func=Distance.manhattan, validation_func=Validation.sse):
        return validation_func(y_truth, Knn.predict(X, y, X_test, K, distance_func))

    @staticmethod
    def impute(dataset, K=3, distance_func=Distance.euclidean):
        dataset = dataset.copy()
        nan_sum = dataset.isna().sum().sort_values()
        for col in nan_sum[nan_sum != 0].index:
            na_mask = dataset[col].isna()
            train = dataset[na_mask == False].dropna(axis=1)
            X_test = dataset.loc[na_mask, train.columns[train.columns != col]]
            X = train.loc[:, train.columns != col]
            y = train[col]
            D = Knn.get_distances(X, y, X_test, distance_func).argsort()[:, :3]
            nan_val = [y.iloc[d].mean() for d in D]
            dataset.loc[na_mask, col] = nan_val
        return dataset

# Knn.validate(*train_test_split(*get_fold(1, dataset)), K=6, validation_func=Validation.accuracy)
# dataset = data_diabetes
# Knn.impute(dataset)
# dataset


class Dwknn(Knn):
    # Distance Weighted K-Nearest Neighbor

    @staticmethod
    def predict(X, y, X_test, K=3, distance_func=Distance.manhattan):
        D = Knn.get_distances(X, y, X_test, distance_func)
        iD_sorted = D.argsort()[:, :K]
        weights = 1 / np.array([D[i].take(iD_sorted[i])
                               for i in range(D.shape[0])])
        knn = np.array([y.iloc[ids][:K] for ids in iD_sorted])
        return (((1-knn)*weights).sum(1) < (knn*weights).sum(1)).astype(int)

    @staticmethod
    def validate(X, y, X_test, y_truth, K=3, distance_func=Distance.manhattan, validation_func=Validation.sse):
        return validation_func(y_truth, Dwknn.predict(X, y, X_test, K, distance_func))

# X, y, X_test, y_true = train_test_split(*get_fold(4, dataset))
# display(y_true.loc[:3])
# Dwknn.predict(X, y, X_test.loc[:3], K=5)
# conf = Dwknn.validate(X, y, X_test, y_true, validation_func=Validation.confusion_matrix)
# print(conf)
# get_scores(conf)
