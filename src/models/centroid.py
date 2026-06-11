"""Nearest-centroid classifier: each subject is summarized by its mean vector."""

import numpy as np

from .base import Model


class NearestCentroid(Model):
    METRICS = ["euclidean", "manhattan", "scaled-manhattan"]
    param_grid = {"metric": METRICS}

    def __init__(self, metric="scaled-manhattan"):
        assert metric in NearestCentroid.METRICS
        self.metric = metric

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        self.mus = x.mean(axis=0)
        self.sigmas = np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
        x = (x - self.mus) / self.sigmas
        self.subjects = np.unique(y)
        self.centroids = np.array([x[y == s].mean(axis=0) for s in self.subjects])
        spreads = np.array(
            [
                np.abs(x[y == s] - c).mean(axis=0)
                for s, c in zip(self.subjects, self.centroids)
            ]
        )
        self.spreads = np.where(spreads == 0, 1, spreads)
        return self

    def predict(self, x):
        x = (x - self.mus) / self.sigmas
        if self.metric == "euclidean":
            distances = np.sqrt(np.sum((self.centroids - x) ** 2, axis=1))
        elif self.metric == "manhattan":
            distances = np.sum(np.abs(self.centroids - x), axis=1)
        elif self.metric == "scaled-manhattan":
            distances = np.sum(np.abs(self.centroids - x) / self.spreads, axis=1)
        else:
            raise Exception("Unknown metric")

        return self.subjects[distances.argmin()]
