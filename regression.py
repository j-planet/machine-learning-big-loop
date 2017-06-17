from sklearn.linear_model import LinearRegression

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
     {})
]




x, y = gen_reg_data(10, 2, 100, 5, sum, 1)

big_loop()