"""
parameter settings used by multiple classifiers/regressors
"""

import numpy as np

n_estimators = {'n_estimators': [2, 3, 5, 10, 25]}
max_features = {'max_features': [3, 5, 7, 'auto', 'log2', None]}
penalty_12 = {'penalty': ['l1', 'l2']}
penalty_12none = {'penalty': ['l1', 'l2', None]}
penalty_12e = {'penalty': ['l1', 'l2', 'elasticnet']}
penalty_all = {'penalty': ['l1', 'l2', None, 'elasticnet']}
max_iter = {'max_iter': [100, 300, 500, 1000]}
max_iter_inf = {'max_iter': [100, 300, 500, 1000, np.inf]}
max_iter_inf2 = {'max_iter': [100, 300, 500, 1000, -1]}
tol = {'tol': [1e-4, 1e-3, 1e-2]}
warm_start = {'warm_start': [True, False]}
alpha = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]}
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