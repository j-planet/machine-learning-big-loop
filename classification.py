import numpy as np
from sklearn import datasets
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

from utilities import *


def gen_classification_data(n=None):
    """
    uses the iris data
    :return: x, y
    """

    iris = datasets.load_iris()
    x = iris.data
    y = iris.target

    if n:
        half = int(n/2)
        np.concatenate((x[:half], x[-half:]), 1), np.concatenate((y[:half], y[-half:]), 0)

    return x, y

models_n_params = [
    (linear_model.SGDClassifier,
     {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
      'alpha': [0.0001, 0.001, 0.1],
      'penalty': ['l1', 'l2', 'none']}),

    # (RandomForestClassifier,
    #  {'n_estimators': [2, 3, 5, 10, 25],
    #   'max_features': [3, 5, 7, 'auto', 'log2'],
    #   'min_samples_leaf': [2, 3]}),
    #
    # (GradientBoostingClassifier,
    #  {'n_estimators': [3, 5, 10, 25],
    #   'max_depth': [2, 3, 4],
    #   'max_features': [2, 'auto', 'log2', None]}),
    #
    # (KMeans,
    #  {'n_clusters': [2],
    #   'algorithm': ['auto', 'full', 'elkan'],
    #   'init': ['k-means++', 'random']}),
    #
    # (KNeighborsClassifier,
    #  {'n_neighbors': [5, 7, 10, 15, 20],
    #   'weights': ['uniform', 'distance'],
    #   'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    #   'leaf_size': [2, 3, 5, 10],
    #   'p': [1, 2]})
]

x, y = gen_classification_data()
big_loop(models_n_params, x, y, isClassification=True, verbose=True)