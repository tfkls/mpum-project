import numpy as np

from .base import Model


class KNN(Model):
    METRICS = ["euclidean", "cosine", "manhattan", "chebyshev", "mahalanobis"]
    param_grid = {"k": [1, 3, 5, 7, 9], "metric": METRICS}

    def __init__(self, k=3, metric="euclidean"):
        assert metric in KNN.METRICS
        self.metric = metric
        self.k = k

    def train(self, training_data):
        self.x = training_data.drop(columns=["subject"]).to_numpy()
        self.y = training_data["subject"].to_numpy()
        self.mus = self.x.mean(axis=0)
        self.sigmas = self.x.std(axis=0)
        self.sigmas = np.where(self.sigmas == 0, 1, self.sigmas)
        self.x = (self.x - self.mus) / self.sigmas
        if self.metric == "mahalanobis":
            covariance_matrix = np.cov(self.x, rowvar=False)
            self.cov_inv = np.linalg.inv(covariance_matrix)
        return self

    def _distances(self, x):
        x = (x - self.mus) / self.sigmas
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((self.x - x) ** 2, axis=1))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.x - x), axis=1)
        elif self.metric == "cosine":
            distances = 1 - np.dot(self.x, x) / (
                np.linalg.norm(x) * np.linalg.norm(self.x, axis=1)
            )
        elif self.metric == "chebyshev":
            distances = np.max(np.abs(self.x - x), axis=1)
        elif self.metric == "mahalanobis":
            diff = self.x - x
            distances = np.sqrt(np.sum((diff @ self.cov_inv) * diff, axis=1))
        else:
            raise Exception("Unknown metric")
        return distances

    def predict(self, x):
        nearest_indices = np.argsort(self._distances(x))[: self.k]
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

        return int(result if result is not None else -1)


class KNNWeighted(KNN):
    param_grid = {
        "k": [3, 5, 9, 15],
        "metric": ["euclidean", "manhattan", "chebyshev"],
        "alpha": [0.33, 0.5, 1.0],
    }

    def __init__(self, k=5, metric="manhattan", alpha=1.0):
        super().__init__(k, metric)
        self.alpha = alpha

    def train(self, training_data):
        super().train(training_data)
        counts = np.bincount(self.y).astype(float)
        self.vote_weights = np.where(counts > 0, counts, 1) ** -self.alpha
        return self

    def predict(self, x):
        nearest = self.y[np.argsort(self._distances(x))[: self.k]]
        scores = np.bincount(nearest, weights=self.vote_weights[nearest])
        return int(scores.argmax())
