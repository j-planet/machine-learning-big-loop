from sklearn.linear_model import LinearRegression, Ridge

from utilities import *


def gen_reg_data(x_mu=10, x_sigma=1, num_samples=100, num_features=3,
                 y_formula=sum, y_sigma=1):
    """
    generate some fake data for us to work with
    :return: x, y
    """
    x = np.random.normal(x_mu, x_sigma, (num_samples, num_features))
    y = np.apply_along_axis(y_formula, 1, x) + np.random.normal(0, y_sigma, (num_samples,))

    return x, y


models_n_params = [
    (LinearRegression,
    {'fit_intercept': [True],
     'normalize': [True, False],
     'copy_X': [True]}),

    (Ridge,
     {'alpha': [1e-3, 1e-2, 0.1, 1, 2, 10],
      'normalize': [True, False],
      'tol': [1e-4, 1e-3, 1e-2],
      'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
      'copy_X': [True]
      })
]


x, y = gen_reg_data(10, 2, 100, 5, sum, 1)
reg = LinearRegression(normalize=False)
reg.fit(x,y)

big_loop(models_n_params, x, y, isClassification=False)