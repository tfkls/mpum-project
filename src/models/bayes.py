import numpy as np

from .base import Model


class NaiveBayes(Model):
    param_grid = {"alpha": [0, 1]}

    def __init__(self):
        pass

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()

        self.classes = np.unique(y)
        self.means = {}
        self.variances = {}
        self.counts = {}

        for c in self.classes:
            xc = x[y == c]
            self.counts[c] = xc.shape[0]
            self.means[c] = xc.mean(axis=0)
            self.variances[c] = xc.var(axis=0, ddof=1)

        return self

    def predict(self, x):
        result = None
        best_score = -np.inf
        for c in self.classes:
            m = self.means[c]
            v = self.variances[c]
            score = np.sum(
                -0.5 * np.log(2 * np.pi * v) - ((x - m) ** 2) / (2 * v)
            ) + self.alpha * np.log(self.counts[c])

            if score > best_score:
                best_score = score
                result = c

        return result
