import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, \
    Perceptron, PassiveAggressiveClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from utilities import *
from universal_params import *


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

linear_models_n_params = [
    (SGDClassifier,
     {'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
      'alpha': [0.0001, 0.001, 0.1],
      **penalty_12none
      }),

    (RandomForestClassifier,
     {**max_features, **n_estimators,
      'min_samples_leaf': [2, 3]}),

    (GradientBoostingClassifier,
     {**max_features, **n_estimators,
      'max_depth': [2, 3, 4]
      }),

    (LogisticRegression,
     {**penalty_12, **max_iter, **tol, ** warm_start, **C,
      'solver': ['liblinear']
      }),

    (Perceptron,
     {**penalty_all, **alpha, **n_iter, **eta0, **warm_start
      }),

    (PassiveAggressiveClassifier,
     {**C, **n_iter, **warm_start,
      'loss': ['hinge', 'squared_hinge'],
      })
]

svm_models_n_params = [
    (SVC,
     {**C, **kernel, **degree, **gamma, **coef0, **shrinking, **tol, **max_iter_inf2}),

    (NuSVC,
     {**nu, **kernel, **degree, **gamma, **coef0, **shrinking, **tol
      }),

    (LinearSVC,
     { **C, **penalty_12, **tol, **max_iter,
       'loss': ['hinge', 'squared_hinge'],
       })
]


neighbor_models_n_params = [

    (KMeans,
     {'algorithm': ['auto', 'full', 'elkan'],
      'init': ['k-means++', 'random']}),

    (KNeighborsClassifier,
     {**n_neighbors, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2]
      }),

    (NearestCentroid,
     {**neighbor_metric,
      'shrink_threshold': [1e-3, 1e-2, 0.1, 0.5, 0.9, 2]
      })

]

x, y = gen_classification_data()
big_loop(svm_models_n_params,
         StandardScaler().fit_transform(x), y, isClassification=True)