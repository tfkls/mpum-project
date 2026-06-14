from .centroid import NearestCentroid
from .decision_tree import DecisionTree
from .knn import KNN, KNNWeighted
from .mlp import MLP
from .random_forest import RandomForest
from .triplet import FakeMLPEmbedding, MLPEmbedding, TripletEmbedding

MODELS = [
    FakeMLPEmbedding,
    KNN,
    KNNWeighted,
    MLP,
    DecisionTree,
    RandomForest,
    NearestCentroid,
    TripletEmbedding,
]
