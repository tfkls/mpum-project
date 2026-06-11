"""Multi-layer perceptron (sklearn-backed)."""

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from .base import Model


class MLP(Model):
    param_grid = {"hidden_layer_sizes": [(64,), (64, 64), (128, 64)]}

    def __init__(self, hidden_layer_sizes=(64, 64)):
        self.hidden_layer_sizes = hidden_layer_sizes

    def train(self, training_data):
        x = training_data.drop(columns=["subject"]).to_numpy()
        y = training_data["subject"].to_numpy()
        self.scaler = StandardScaler()
        x = self.scaler.fit_transform(x)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=400,
            random_state=1,
            activation="relu",
        )
        self.mlp.fit(x, y)
        return self

    def predict(self, x):
        x = self.scaler.transform(x.reshape(1, -1))
        return self.mlp.predict(x)[0]
