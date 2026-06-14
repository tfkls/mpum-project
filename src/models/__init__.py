from .bayes import NaiveBayes
from .centroid import NearestCentroid
from .decision_tree import DecisionTree
from .knn import KNN, KNNWeighted
from .mlp import MLP
from .random_forest import RandomForest
from .triplet import MLPEmbedding, PCAEmbedding, TripletEmbedding

MODELS = [
    KNN,
    KNNWeighted,
    MLP,
    DecisionTree,
    RandomForest,
    NearestCentroid,
    NaiveBayes,
    TripletEmbedding,
    MLPEmbedding,
    PCAEmbedding,
]
