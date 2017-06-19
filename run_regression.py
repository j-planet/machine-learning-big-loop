import warnings
warnings.filterwarnings('ignore')
from multiprocessing import cpu_count

# linear models: http://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, ElasticNet, \
    Lars, LassoLars, \
    OrthogonalMatchingPursuit, \
    BayesianRidge, ARDRegression, \
    SGDRegressor, \
    PassiveAggressiveRegressor, \
    RANSACRegressor, HuberRegressor

from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler

# svm models: http://scikit-learn.org/stable/modules/svm.html
from sklearn.svm import SVR, NuSVR, LinearSVR

# neighbor models: http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.RadiusNeighborsRegressor.html#sklearn.neighbors.RadiusNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor, KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from utilities import *
from universal_params import *


def gen_reg_data(x_mu=10., x_sigma=1., num_samples=100, num_features=3,
                 y_formula=sum, y_sigma=1.):
    """
    generate some fake data for us to work with
    :return: x, y
    """
    x = np.random.normal(x_mu, x_sigma, (num_samples, num_features))
    y = np.apply_along_axis(y_formula, 1, x) + np.random.normal(0, y_sigma, (num_samples,))

    return x, y


linear_models_n_params = [
    (LinearRegression, normalize),

    (Ridge,
     {**alpha, **normalize, **tol,
      'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
      }),

    (Lasso,
     {**alpha, **normalize, **tol, **warm_start
      }),

    (ElasticNet,
     {**alpha, **normalize, **tol,
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      }),

    (Lars,
     {**normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     {**normalize, **max_iter_inf, **normalize, **alpha
      }),

    (OrthogonalMatchingPursuit,
     {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      **tol, **normalize
      }),

    (BayesianRidge,
     {
         'n_iter': [100, 300, 1000],
         **tol, **normalize,
         'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
         'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
     }),

    # WARNING: ARDRegression takes a long time to run
    (ARDRegression,
     {'n_iter': [100, 300, 1000],
      **tol, **normalize,
      'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
      'threshold_lambda': [1e2, 1e3, 1e4, 1e6]}),

    (SGDRegressor,
     {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
      **penalty_12e, **n_iter, **epsilon, **eta0,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      'learning_rate': ['constant', 'optimal', 'invscaling'],
      'power_t': [0.1, 0.25, 0.5]
      }),

    (PassiveAggressiveRegressor,
     {**C, **epsilon, **n_iter, **warm_start,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive']
      }),

    (RANSACRegressor,
     {'min_samples': [0.1, 0.5, 0.9, None],
      'max_trials': n_iter['n_iter'],
      'stop_score': [0.8, 0.9, 1],
      'stop_probability': [0.9, 0.95, 0.99, 1],
      'loss': ['absolute_loss', 'squared_loss']
      }),

    (HuberRegressor,
     { 'epsilon': [1.1, 1.35, 1.5, 2],
       **max_iter, **alpha, **warm_start, **tol
       }),

    (KernelRidge,
     {**alpha, **degree, **gamma, **coef0
      })
]

linear_models_n_params_small = [
    (LinearRegression, normalize),

    (Ridge,
     {**alpha_small, **normalize
      }),

    (Lasso,
     {**alpha_small, **normalize
      }),

    (ElasticNet,
     {**alpha, **normalize,
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
      }),

    (Lars,
     {**normalize,
      'n_nonzero_coefs': [100, 300, 500, np.inf],
      }),

    (LassoLars,
     {**normalize, **max_iter_inf, **normalize, **alpha_small
      }),

    (OrthogonalMatchingPursuit,
     {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
      **normalize
      }),

    (BayesianRidge,
     { 'n_iter': [100, 300, 1000],
       'alpha_1': [1e-6, 1e-3],
       'alpha_2': [1e-6, 1e-3],
       'lambda_1': [1e-6, 1e-3],
       'lambda_2': [1e-6, 1e-3],
       **normalize,
       }),

    # WARNING: ARDRegression takes a long time to run
    (ARDRegression,
     {'n_iter': [100, 300],
      **normalize,
      'alpha_1': [1e-6, 1e-3],
      'alpha_2': [1e-6, 1e-3],
      'lambda_1': [1e-6, 1e-3],
      'lambda_2': [1e-6, 1e-3],
      }),

    (SGDRegressor,
     {'loss': ['squared_loss', 'huber'],
      **penalty_12e, **n_iter,
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
      }),

    (PassiveAggressiveRegressor,
     {**C, **n_iter,
      }),

    (RANSACRegressor,
     {'min_samples': [0.1, 0.5, 0.9, None],
      'max_trials': n_iter['n_iter'],
      'stop_score': [0.8, 1],
      'loss': ['absolute_loss', 'squared_loss']
      }),

    (HuberRegressor,
     { **max_iter, **alpha_small,
       }),

    (KernelRidge,
     {**alpha_small, **degree,
      })
]

svm_models_n_params_small = [
    (SVR,
     {**kernel, **degree, **shrinking
      }),

    (NuSVR,
     {**nu_small, **kernel, **degree, **shrinking,
      }),

    (LinearSVR,
     {**C_small, **epsilon,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 1, 10]
      })
]

svm_models_n_params = [
    (SVR,
     {**C, **epsilon, **kernel, **degree, **gamma, **coef0, **shrinking, **tol, **max_iter_inf2
      }),

    (NuSVR,
     {**C, **nu, **kernel, **degree, **gamma, **coef0, **shrinking , **tol, **max_iter_inf2
      }),

    (LinearSVR,
     {**C, **epsilon, **tol, **max_iter,
      'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
      'intercept_scaling': [0.1, 0.5, 1, 5, 10]
      })
]

neighbor_models_n_params = [
    (RadiusNeighborsRegressor,
     {**neighbor_radius, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'weights': ['uniform', 'distance'],
      'p': [1, 2],
      }),

    (KNeighborsRegressor,
     {**n_neighbors, **neighbor_algo, **neighbor_leaf_size, **neighbor_metric,
      'p': [1, 2],
      'weights': ['uniform', 'distance'],
      })
]

gaussianprocess_models_n_params = [
    (GaussianProcessRegressor,
     {'kernel': [RBF(), ConstantKernel(), DotProduct(), WhiteKernel()],
      'n_restarts_optimizer': [3],
      'alpha': [1e-10, 1e-5],
      'normalize_y': [True, False]
      })
]

nn_models_n_params = [
    (MLPRegressor,
     { 'hidden_layer_sizes': [(16,), (64,), (100,), (32, 64)],
       'activation': ['identity', 'logistic', 'tanh', 'relu'],
       **alpha, **learning_rate, **tol, **warm_start,
       'batch_size': ['auto', 50],
       'max_iter': [1000],
       'early_stopping': [True, False],
       'epsilon': [1e-8, 1e-5]
       })
]

nn_models_n_params_small = [
    (MLPRegressor,
     { 'hidden_layer_sizes': [(64,), (32, 64)],
       'activation': ['identity', 'tanh', 'relu'],
       'max_iter': [500],
       'early_stopping': [True],
       **learning_rate_small
       })
]

tree_models_n_params = [

    (DecisionTreeRegressor,
     {**max_features, **max_depth, **min_samples_split, **min_samples_leaf, **min_impurity_split,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {**n_estimators, **max_features, **max_depth, **min_samples_split,
      **min_samples_leaf, **min_impurity_split, **warm_start,
      'criterion': ['mse', 'mae']}),

]

tree_models_n_params_small = [
    (DecisionTreeRegressor,
     {**max_features_small, **max_depth_small, **min_samples_split, **min_samples_leaf,
      'criterion': ['mse', 'mae']}),

    (ExtraTreesRegressor,
     {**n_estimators_small, **max_features_small, **max_depth_small, **min_samples_split,
      **min_samples_leaf,
      'criterion': ['mse', 'mae']})
]

def run_linear_models(x, y, small = True, normalize_x = True):
    return big_loop(linear_models_n_params_small if small else linear_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_svm_models(x, y, small = True, normalize_x = True):
    return big_loop(svm_models_n_params_small if small else svm_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_neighbor_models(x, y, normalize_x = True):
    return big_loop(neighbor_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_gaussian_models(x, y, normalize_x = True):
    return big_loop(gaussianprocess_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_nn_models(x, y, small = True, normalize_x = True):
    return big_loop(nn_models_n_params_small if small else nn_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_tree_models(x, y, small = True, normalize_x = True):
    return big_loop(tree_models_n_params_small if small else tree_models_n_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y, isClassification=False)

def run_all(x, y, small = True, normalize_x = True, n_jobs=cpu_count()-1):

    all_params = (linear_models_n_params_small if small else linear_models_n_params) + \
                 (nn_models_n_params_small if small else nn_models_n_params) + \
                 ([] if small else gaussianprocess_models_n_params) + \
                 neighbor_models_n_params + \
                 (svm_models_n_params_small if small else svm_models_n_params) + \
                 (tree_models_n_params_small if small else tree_models_n_params)

    return big_loop(all_params,
                    StandardScaler().fit_transform(x) if normalize_x else x, y,
                    isClassification=False, n_jobs=n_jobs)


if __name__ == '__main__':

    x, y = gen_reg_data(10, 3, 100, 3, sum, 0.3)
    run_all(x, y, small=True, normalize_x=True)
