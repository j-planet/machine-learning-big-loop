# linear models:
# http://scikit-learn.org/stable/modules/linear_model.html#stochastic-gradient-descent-sgd
from sklearn.linear_model import \
    LinearRegression, Ridge, Lasso, ElasticNet, \
    Lars, LassoLars, \
    OrthogonalMatchingPursuit, \
    BayesianRidge, ARDRegression, \
    SGDRegressor

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


models_n_params = [
    # (LinearRegression,
    # {'normalize': [True, False]
    #  }),
    #
    # (Ridge,
    #  {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 10],
    #   'normalize': [True, False],
    #   'tol': [1e-4, 1e-3, 1e-2],
    #   'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
    #   }),
    #
    # (Lasso,
    #  {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 10],
    #   'normalize': [True, False],
    #   'tol': [1e-4, 1e-3, 1e-2],
    #   'warm_start': [True, False]
    #   }),
    #
    # (ElasticNet,
    #  {'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 2, 10],
    #   'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
    #   'normalize': [True, False],
    #   'tol': [1e-4, 1e-3, 1e-2]
    #  }),

    # (Lars,
    #  {'normalize': [True, False],
    #   'n_nonzero_coefs': [100, 300, 500, np.inf],
    #   }),

    # (LassoLars,
    #  {'normalize': [True, False],
    #   'max_iter': [100, 300, 500, np.inf],
    #   'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1, 10],
    #   }),
    #
    # (OrthogonalMatchingPursuit,
    #  {'n_nonzero_coefs': [100, 300, 500, np.inf, None],
    #   'tol': [1e-4, 1e-3, 1e-2],
    #   'normalize': [True, False]
    #   }),

    # (BayesianRidge,
    #  {
    #      'n_iter': [100, 300, 1000],
    #      'tol': [1e-4, 1e-3, 1e-2],
    #      'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #      'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #      'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #      'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #      'normalize': [True, False]
    #  }),
    #
    # (ARDRegression,
    #  {'n_iter': [100, 300, 1000],
    #   'tol': [1e-4, 1e-3, 1e-2],
    #   'alpha_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #   'alpha_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #   'lambda_1': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #   'lambda_2': [1e-6, 1e-4, 1e-2, 0.1, 0],
    #   'normalize': [True, False],
    #   'threshold_lambda': [1e2, 1e3, 1e4, 1e6]}),

    (SGDRegressor,
     {'loss': ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
      'penalty': ['l1', 'l2', 'elasticnet'],
      'alpha': [1e-6, 1e-5, 1e-2, 'optimal'],
      'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9],
      **n_iter,
      'epsilon': [1e-3, 1e-2, 0.1, 0],
      'learning_rate': ['constant', 'optimal', 'invscaling'],
      **eta0,
      'power_t': [0.1, 0.25, 0.5]
      })
]


x, y = gen_reg_data(10, 3, 100, 3, sum, 0.3)

big_loop(models_n_params, x, y, isClassification=False)