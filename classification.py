import warnings
warnings.filterwarnings('ignore')
import numpy as np
from sklearn import datasets
from sklearn.linear_model import SGDClassifier, LogisticRegression, \
    Perceptron, PassiveAggressiveClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, Matern, StationaryKernelMixin, WhiteKernel
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

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

svm_models_n_params_small = [
    (SVC,
     {**kernel, **degree, **shrinking
      }),

    (NuSVC,
     {**nu_small, **kernel, **degree, **shrinking
      }),

    (LinearSVC,
     { **C_small,
       'penalty': ['l2'],
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
      }),

    (RadiusNeighborsClassifier,
     {**neighbor_radius, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      'outlier_label': [-1]
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessClassifier,
     {**warm_start,
      'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      # 'kernel': [RBF(), ConstantKernel(), DotProduct(), Matern(), StationaryKernelMixin(), WhiteKernel()],
      'max_iter_predict': max_iter['max_iter'],
      'n_restarts_optimizer': [3],
      })
]

bayes_models_n_params = [
    (GaussianNB, {})
]

nn_models_n_params = [
    (MLPClassifier,
     { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 32)],
       'activation': ['identity', 'logistic', 'tanh', 'relu'],
       **alpha, **learning_rate, **tol, **warm_start,
       'batch_size': ['auto', 50],
       'max_iter': [1000],
       'early_stopping': [True, False],
       'epsilon': [1e-8, 1e-5]
       })
]

tree_models_n_params = [

    (RandomForestClassifier,
     {'criterion': ['gini', 'entropy'],
      **max_features, **n_estimators, **max_depth,
      **min_samples_split, **min_impurity_split, **warm_start, **min_samples_leaf,
      }),

    (DecisionTreeClassifier,
     {'criterion': ['gini', 'entropy'],
      **max_features, **max_depth, **min_samples_split, **min_impurity_split, **min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {**n_estimators, **max_features, **max_depth,
      **min_samples_split, **min_samples_leaf, **min_impurity_split, **warm_start,
      'criterion': ['gini', 'entropy']})
]


tree_models_n_params_small = [

    (RandomForestClassifier,
     {**max_features_small, **n_estimators_small, **min_samples_split, **max_depth_small, **min_samples_leaf
      }),

    (DecisionTreeClassifier,
     {**max_features_small, **max_depth_small, **min_samples_split, **min_samples_leaf
      }),

    (ExtraTreesClassifier,
     {**n_estimators_small, **max_features_small, **max_depth_small,
      **min_samples_split, **min_samples_leaf})
]

x, y = gen_classification_data()
big_loop(tree_models_n_params_small,
         StandardScaler().fit_transform(x), y, isClassification=True)



















