"""
parameter settings used by multiple classifiers/regressors
"""

import numpy as np

penalty_12 = {'penalty': ['l1', 'l2']}
penalty_12none = {'penalty': ['l1', 'l2', None]}
penalty_12e = {'penalty': ['l1', 'l2', 'elasticnet']}
penalty_all = {'penalty': ['l1', 'l2', None, 'elasticnet']}
max_iter = {'max_iter': [100, 300, 1000]}
max_iter_inf = {'max_iter': [100, 300, 500, 1000, np.inf]}
max_iter_inf2 = {'max_iter': [100, 300, 500, 1000, -1]}
tol = {'tol': [1e-4, 1e-3, 1e-2]}
warm_start = {'warm_start': [True, False]}
alpha = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]}
alpha_small = {'alpha': [1e-5, 1e-3, 0.1, 1]}
n_iter = {'n_iter': [5, 10, 20]}

eta0 = {'eta0': [1e-4, 1e-3, 1e-2, 0.1]}
C = {'C': [1e-2, 0.1, 1, 5, 10]}
C_small = {'C': [ 0.1, 1, 5]}
epsilon = {'epsilon': [1e-3, 1e-2, 0.1, 0]}
normalize = {'normalize': [True, False]}
kernel = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
degree = {'degree': [1, 2, 3, 4, 5]}
gamma = {'gamma': list(np.logspace(-9, 3, 6)) + ['auto']}
gamma_small = {'gamma': list(np.logspace(-6, 3, 3)) + ['auto']}
coef0 = {'coef0': [0, 0.1, 0.3, 0.5, 0.7, 1]}
coef0_small = {'coef0': [0, 0.4, 0.7, 1]}
shrinking = {'shrinking': [True, False]}
nu = {'nu': [1e-4, 1e-2, 0.1, 0.3, 0.5, 0.75, 0.9]}
nu_small = {'nu': [1e-2, 0.1, 0.5, 0.9]}

n_neighbors = {'n_neighbors': [5, 7, 10, 15, 20]}
neighbor_algo = {'algorithm': ['ball_tree', 'kd_tree', 'brute']}
neighbor_leaf_size = {'leaf_size': [1, 2, 5, 10, 20, 30, 50, 100]}
neighbor_metric = {'metric': ['cityblock', 'euclidean', 'l1', 'l2', 'manhattan']}
neighbor_radius = {'radius': [1e-2, 0.1, 1, 5, 10]}
learning_rate = {'learning_rate': ['constant', 'invscaling', 'adaptive']}
learning_rate_small = {'learning_rate': ['invscaling', 'adaptive']}

n_estimators = {'n_estimators': [2, 3, 5, 10, 25, 50, 100]}
n_estimators_small = {'n_estimators': [2, 10, 25, 100]}
max_features = {'max_features': [3, 5, 10, 25, 50, 'auto', 'log2', None]}
max_features_small = {'max_features': [3, 5, 10, 'auto', 'log2', None]}
max_depth = {'max_depth': [None, 3, 5, 7, 10]}
max_depth_small = {'max_depth': [None, 5, 10]}
min_samples_split = {'min_samples_split': [2, 5, 10, 0.1]}
min_impurity_split = {'min_impurity_split': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]}
tree_learning_rate = {'learning_rate': [0.8, 1]}
min_samples_leaf = {'min_samples_leaf': [2]}


