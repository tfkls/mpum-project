import numpy as np

from .base import Model
from .decision_tree import DecisionTree


class RandomForest(Model):
    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [15, 10**9],
        "alpha": [0, 0.5, 1.0],
    }

    def __init__(self, n_estimators=100, max_depth=10**9, alpha=0, seed=0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.alpha = alpha
        self.seed = seed

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        rng = np.random.default_rng(self.seed)
        max_features = int(np.ceil(np.sqrt(x.shape[1])))
        self.trees = []
        for _ in range(self.n_estimators):
            rows = rng.integers(0, len(y), len(y))
            tree = DecisionTree(
                max_depth=self.max_depth,
                alpha=self.alpha,
                max_features=max_features,
                seed=int(rng.integers(2**31)),
            )
            self.trees.append(tree.train_arrays(x[rows], y[rows]))
        return self

    def predict(self, x):
        votes = np.bincount([tree.predict(x) for tree in self.trees])
        return int(votes.argmax())
