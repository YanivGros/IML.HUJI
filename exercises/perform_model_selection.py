from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    eps = np.random.normal(0, noise, size=n_samples)
    x = np.random.uniform(-1.2, 2, size=n_samples)
    true_f_of_x = (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noise_f_of_x = true_f_of_x + eps
    train_X, train_y, test_X, test_y = split_train_test(pd.Series(x), pd.Series(noise_f_of_x), 2.0 / 3.0)
    # go.Figure([go.Scatter(name='Train Set', x=train_X, y=train_y, mode='markers',
    #                       marker_color='blue'),
    #            go.Scatter(name='Test Set', x=test_X, y=test_y, mode='markers',
    #                       marker_color='red'),
    #            go.Scatter(name='True Model', x=x, y=true_f_of_x, mode='markers',
    #                       marker=dict(symbol="x"))]) \
    #     .update_layout(title=r"$\text{(1) }\text{Scatter plot of training set, test set and true model}$",
    #                    xaxis_title=r"$\text{x}$",
    #                    yaxis_title=r"$\text{F(x)}$").show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10


    num_fold = 5
    max_polynomial_degree = 10
    # validations = np.array_split(np.indices(pd.Series.to_numpy(train_X), 5))
    validations = np.array_split(train_X.index.to_series(), num_fold)
    print()

    # validations = [(train_X[v.ravel()], train_y[v.ravel()]) for v in validations]

    # Train and evaluate models for all values of k
    # train_errors, test_errors, val_errors = [], [], [[] for _ in range(len(validations))]
    avg_training_loss, avg_valid_lost = [], []

    for degree in range(max_polynomial_degree + 1):
        cur_model_loss_train = []
        cur_model_loss_valid = []
        for v in validations:
            part_train_X, part_train_y = train_X[~v], train_y[~v]
            validations_X, validations_y = train_X[v], train_y[v]
            model = PolynomialFitting(degree).fit(part_train_X, part_train_y)
            cur_model_loss_train.append(model.loss(part_train_X, part_train_y))
            cur_model_loss_valid.append(model.loss(validations_X, validations_y))
        avg_training_loss.append(np.average(cur_model_loss_train))
        avg_valid_lost.append(np.average(cur_model_loss_valid))
    best_parm = np.argmin(avg_valid_lost)
    print()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    raise NotImplementedError()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    raise NotImplementedError()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    raise NotImplementedError()
