"""Triplet metric learning (from the TypeNet paper); NearestCentroid over an embedding (linear or preprocessed via MLP/neural nets)"""

import numpy as np
from sklearn.neural_network import MLPClassifier

from .base import Model
from .mlp import with_bias


class TripletEmbedding(Model):
    param_grid = {"dim": [8, 16], "margin": [1.0, 3.0]}

    def __init__(self, dim=16, margin=1.0, steps=10000, batch=256, lr=0.01, seed=0):
        self.dim = dim
        self.margin = margin
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.seed = seed

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        self.mus = x.mean(axis=0)
        self.sigmas = np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
        x = (x - self.mus) / self.sigmas

        rows_of = {s: np.flatnonzero(y == s) for s in np.unique(y)}
        self._fit_embedding(x, y, rows_of)

        embedded = self._embed(x)
        self.subjects = np.array(sorted(rows_of))
        self.centroids = np.array(
            [embedded[rows_of[s]].mean(axis=0) for s in self.subjects]
        )
        return self

    def _fit_embedding(self, x, y, rows_of):
        rng = np.random.default_rng(self.seed)
        others_of = {s: np.flatnonzero(y != s) for s in rows_of}
        self.w = rng.normal(size=(self.dim, x.shape[1])) / np.sqrt(x.shape[1])
        for _ in range(self.steps):
            anchor = rng.integers(0, len(y), self.batch)
            positive = np.array([rng.choice(rows_of[subject]) for subject in y[anchor]])
            negative = np.array(
                [rng.choice(others_of[subject]) for subject in y[anchor]]
            )

            pos_dist = x[anchor] - x[positive]
            neg_dist = x[anchor] - x[negative]
            loss = (
                self.margin
                + ((pos_dist @ self.w.T) ** 2).sum(axis=1)
                - ((neg_dist @ self.w.T) ** 2).sum(axis=1)
            )

            pos_act, neg_act = pos_dist[loss > 0], neg_dist[loss > 0]
            gradient = (
                2 * self.w @ (pos_act.T @ pos_act - neg_act.T @ neg_act) / self.batch
            )
            self.w -= self.lr * gradient

    def _embed(self, z):
        return z @ self.w.T

    def predict(self, x):
        e = self._embed((x - self.mus) / self.sigmas)
        distances = ((self.centroids - e) ** 2).sum(axis=1)
        return int(self.subjects[distances.argmin()])


class MLPEmbedding(TripletEmbedding):
    param_grid = {
        "dim": [8, 16],
        "margin": [1.0, 3.0],
        "hidden": [64, 128],
    }

    def __init__(
        self, dim=16, hidden=64, margin=1.0, steps=5000, batch=256, lr=0.03, seed=1
    ):
        super().__init__(dim, margin, steps, batch, lr, seed)
        self.hidden = hidden

    def _fit_embedding(self, x, y, rows_of):
        others_of = {s: np.flatnonzero(y != s) for s in rows_of}

        rng = np.random.default_rng(self.seed)
        x = with_bias(x)
        self.w1 = rng.normal(size=(self.hidden, x.shape[1])) * np.sqrt(2 / x.shape[1])
        self.w2 = rng.normal(size=(self.dim, self.hidden + 1)) * np.sqrt(
            2 / self.hidden
        )

        for _ in range(self.steps):
            anchor = rng.integers(0, len(y), self.batch)
            positive = np.array([rng.choice(rows_of[subject]) for subject in y[anchor]])
            negative = np.array(
                [rng.choice(others_of[subject]) for subject in y[anchor]]
            )

            inputs = x[np.concatenate([anchor, positive, negative])]
            hidden = np.maximum(inputs @ self.w1.T, 0)
            ea, ep, en = np.split(with_bias(hidden) @ self.w2.T, 3)

            loss = (
                self.margin
                + ((ea - ep) ** 2).sum(axis=1)
                - ((ea - en) ** 2).sum(axis=1)
            )
            active = (loss > 0)[:, None]
            pos_dist = 2 * (ea - ep) * active
            neg_dist = 2 * (ea - en) * active

            output_grads = (
                np.concatenate([pos_dist - neg_dist, -pos_dist, neg_dist]) / self.batch
            )
            hidden_grads = (output_grads @ self.w2)[:, :-1] * (hidden > 0)

            self.w2 -= self.lr * output_grads.T @ with_bias(hidden)
            self.w1 -= self.lr * hidden_grads.T @ inputs

    def _embed(self, z):
        hidden = np.maximum(with_bias(z) @ self.w1.T, 0)
        return with_bias(hidden) @ self.w2.T


class FakeMLPEmbedding(TripletEmbedding):
    param_grid = {"hidden_layer_sizes": [(64,), (64, 64), (128, 64)]}

    def __init__(self, hidden_layer_sizes=(128, 64)):
        self.hidden_layer_sizes = hidden_layer_sizes

    def _fit_embedding(self, x, y, rows_of):
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=1000,
            random_state=1,
            activation="relu",
        ).fit(x, y)

    def _embed(self, z):
        for w, b in zip(self.mlp.coefs_[:-1], self.mlp.intercepts_[:-1]):
            z = np.maximum(z @ w + b, 0)
        return z
