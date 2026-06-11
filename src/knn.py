import pandas as pd
import numpy as np

import data_loader


class KNN:
    METRICS = {
        'euclidean',
        'cosine',
        'manhattan',
        'chebyshev',
        'mahalanobis',
    }
    def __init__(self, training_data, k=3, metric='euclidean'):
        assert(metric in KNN.METRICS)
        self.metric = metric
        self.k = k
        self.x = training_data.drop(columns=["subject"]).to_numpy()
        self.y = training_data["subject"].to_numpy()
        self.mus = self.x.mean(axis=0)
        self.sigmas = self.x.std(axis=0)
        self.sigmas = np.where(self.sigmas == 0, 1, self.sigmas)
        self.x = (self.x - self.mus)/self.sigmas
        if metric == 'mahalanobis':
            covariance_matrix = np.cov(self.x, rowvar=False)
            self.cov_inv = np.linalg.inv(covariance_matrix)

    def predict(self, x):
        x = (x - self.mus)/self.sigmas
        if self.metric == 'euclidean':
            distances = np.sqrt(np.sum((self.x - x) ** 2, axis=1))
        elif self.metric == 'manhattan':
            distances = np.sum(np.abs(self.x - x), axis=1)
        elif self.metric == 'cosine':
            distances = 1 - np.dot(self.x, x) / (np.linalg.norm(x) * np.linalg.norm(self.x, axis=1))
        elif self.metric == 'chebyshev':
            distances = np.max(np.abs(self.x - x), axis=1)
        elif self.metric == 'mahalanobis':
            diff = self.x - x
            distances = np.sqrt(np.sum((diff @ self.cov_inv) * diff, axis=1))
        else:
            raise Exception("Unknown metric")

        nearest_indices = np.argsort(distances)[:self.k]
        nearest = self.y[nearest_indices]

        counts = np.bincount(nearest)
        results = np.where(counts == counts.max())[0]

        result = None
        if len(results) == 1:
            result = results[0]
        else:
            for x in nearest:
                if x in results:
                    result = x
                    break

        return result


if __name__ == '__main__':
    data = data_loader.load_data(include_rep=False)
    df_train, df_test = data_loader.split_data(data, sample=[(30, 360), (21, 40)], test_bound=40)
    knn = KNN(df_train, metric='mahalanobis', k=5)
    X_test = df_test.drop(columns=["subject"]).to_numpy()
    Y_test = df_test["subject"].to_numpy()
    for x, y in zip(X_test, Y_test):
        print(f"Expected: {y}, Got: {knn.predict(x)}")