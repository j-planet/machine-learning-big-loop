"""
parameter settings used by multiple classifiers/regressors
"""

n_estimators = {'n_estimators': [2, 3, 5, 10, 25]}
max_features = {'max_features': [3, 5, 7, 'auto', 'log2', None]}
penalty_12 = {'penalty': ['l1', 'l2']}
penalty_12none = {'penalty': ['l1', 'l2', None]}
penalty_all = {'penalty': ['l1', 'l2', None, 'elasticnet']}
max_iter = {'max_iter': [100, 300, 500, 1000]}
tol = {'tol': [1e-4, 1e-3, 1e-2]}
warm_start = {'warm_start': [True, False]}
alpha = {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]}
n_iter = {'n_iter': [5, 10, 20]}
eta0 = {'eta0': [1e-4, 1e-3, 1e-2, 0.1]}