from abc import ABC, abstractmethod


class Model(ABC):
    param_grid: dict[str, list] = {}

    @abstractmethod
    def train(self, training_data) -> "Model":
        """Train the model on the given training DataFrame; returns self."""

    @abstractmethod
    def predict(self, x) -> int:
        """Predict the subject for a single feature row."""

    def __call__(self, x):
        return self.predict(x)
