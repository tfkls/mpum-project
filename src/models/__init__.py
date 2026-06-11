from .decision_tree import DecisionTree
from .knn import KNN, KNNWeighted
from .mlp import MLP
from .random_forest import RandomForest

MODELS = [KNN, KNNWeighted, MLP, DecisionTree, RandomForest]
