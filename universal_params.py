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
tol = {'tol': [1e-4, 1e-3, 1e-2]}
warm_start = {'warm_start': [True, False]}
alpha = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 3, 10]}
n_iter = {'n_iter': [5, 10, 20]}
eta0 = {'eta0': [1e-4, 1e-3, 1e-2, 0.1]}
C = {'C': [1e-2, 0.1, 1, 10, 1e2]}
epsilon = {'epsilon': [1e-3, 1e-2, 0.1, 0]}
normalize = {'normalize': [True, False]}