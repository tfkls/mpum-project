"""Decision tree (CART): greedy axis-aligned splits minimizing weighted Gini."""

import numpy as np

from .base import Model


def gini(counts):
    p = counts / counts.sum(axis=-1, keepdims=True)
    return 1 - (p**2).sum(axis=-1)


class DecisionTree(Model):
    param_grid = {
        "max_depth": [10, 15, 20, 10**9],
        "min_samples": [2, 4, 6, 10],
        "alpha": [0, 0.5, 1.0],
    }

    def __init__(
        self,
        max_depth=10**9,
        min_samples=3,
        alpha=0,
        max_features=10**9,
        seed=0,
    ):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.alpha = alpha
        self.max_features = max_features
        self.seed = seed

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        return self.train_arrays(x, y)

    def train_arrays(self, x, y):
        """NOTE: Labels must be small non-negative ints or we crash :)"""
        w = np.ones(len(y))
        w = w * np.bincount(y, weights=w)[y] ** -self.alpha
        self.rng = np.random.default_rng(self.seed)
        m = np.zeros((len(y), y.max() + 1))
        m[np.arange(len(y)), y] = w
        self.root = self._build(x, m, np.arange(len(y)), depth=0)
        return self

    def _build(self, x, m, rows, depth):
        counts = m[rows].sum(axis=0)
        split = None
        if (
            depth < self.max_depth
            and gini(counts) > 0
            and len(rows) >= self.min_samples
        ):
            split = self._best_split(x, m, rows)
        if split is None:
            return counts.argmax()
        feature, threshold, below, above = split
        return (
            feature,
            threshold,
            self._build(x, m, below, depth + 1),
            self._build(x, m, above, depth + 1),
        )

    def _best_split(self, x, m, rows):
        """Returns: None or (feature, threshold, items <= threshold, items > threshold)"""
        counts = m[rows].sum(axis=0)
        n_features = x.shape[1]
        features = self.rng.choice(
            n_features, min(n_features, self.max_features), replace=False
        )
        best, best_score = None, gini(counts)
        for feature in features:
            order = rows[np.argsort(x[rows, feature])]
            values = x[order, feature]

            # NOTE: was as simple loop before, way too slow (julia is better in that regard ;)
            below = np.cumsum(m[order], axis=0)[:-1]
            above = counts - below
            scores = (
                below.sum(axis=1) * gini(below) + above.sum(axis=1) * gini(above)
            ) / counts.sum()

            legal = values[:-1] != values[1:]
            scores[~legal] = np.inf

            i = scores.argmin()
            if scores[i] < best_score:
                best_score = scores[i]
                best = (
                    feature,
                    (values[i] + values[i + 1]) / 2,
                    order[: i + 1],
                    order[i + 1 :],
                )
        return best

    def predict(self, x):
        node = self.root
        while isinstance(node, tuple):
            feature, threshold, below, above = node
            node = below if x[feature] <= threshold else above
        return node
