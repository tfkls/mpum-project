import numpy as np

from .base import Model


def with_bias(z):
    return np.append(z, np.ones((*z.shape[:-1], 1)), axis=-1)


class MLP(Model):
    param_grid = {"hidden": [64, 128], "alpha": [0, 0.5, 1.0]}

    def __init__(self, hidden=64, alpha=0, steps=10000, batch=256, lr=0.02, seed=0):
        self.hidden = hidden
        self.alpha = alpha
        self.steps = steps
        self.batch = batch
        self.lr = lr
        self.seed = seed

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        self.mus = x.mean(axis=0)
        self.sigmas = np.where(x.std(axis=0) == 0, 1, x.std(axis=0))
        x = with_bias((x - self.mus) / self.sigmas)

        counts = np.bincount(y).astype(float)
        weights = np.where(counts > 0, counts, 1) ** -self.alpha
        weights = weights * len(y) / weights[y].sum()
        ident = np.eye(len(counts))

        rng = np.random.default_rng(self.seed)
        self.w1 = rng.normal(size=(self.hidden, x.shape[1])) * np.sqrt(1 / x.shape[1])
        self.w2 = rng.normal(size=(len(counts), self.hidden + 1)) * np.sqrt(
            1 / self.hidden
        )

        for _ in range(self.steps):
            rows = rng.integers(0, len(y), self.batch)
            hidden = np.maximum(x[rows] @ self.w1.T, 0)
            scores = with_bias(hidden) @ self.w2.T
            exps = np.exp(scores - scores.max(axis=1, keepdims=True))
            softmax = exps / exps.sum(axis=1, keepdims=True)

            output_grads = (
                (softmax - ident[y[rows]]) * weights[y[rows]][:, None] / self.batch
            )
            hidden_grads = (output_grads @ self.w2)[:, :-1] * (hidden > 0)

            self.w2 -= self.lr * output_grads.T @ with_bias(hidden)
            self.w1 -= self.lr * hidden_grads.T @ x[rows]
        return self

    def predict(self, x):
        hidden = np.maximum(with_bias((x - self.mus) / self.sigmas) @ self.w1.T, 0)
        return int((with_bias(hidden) @ self.w2.T).argmax())
