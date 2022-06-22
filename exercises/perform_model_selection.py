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

    avg_training_loss, avg_valid_lost = [], []

    for degree in range(0, 11):
        train_loss, valid_lost = cross_validate(PolynomialFitting(degree),
                                                train_X.to_numpy().flatten(),
                                                train_y.to_numpy().flatten(),
                                                mean_square_error)
        avg_training_loss.append(train_loss)
        avg_valid_lost.append(valid_lost)

    # go.Figure([go.Scatter(name='Avg loss over training set', x=np.arange(0, 11), y=avg_training_loss,
    #                       mode='markers +lines',
    #                       marker_color='blue'),
    #            go.Scatter(name='Avg loss over validation set', x=np.arange(0, 11), y=avg_valid_lost,
    #                       mode='markers +lines',
    #                       marker_color='red')]) \
    #     .update_layout(title=r"$\text{Average loss of polynomial fitting estimator over training and validation set "
    #                          r"as function of the degree of the polynomial}$",
    #                    xaxis_title=r"$\text{Degree of polynomial}$",
    #                    yaxis_title=r"$\text{MSE}$").show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    min_mse_degree = int(np.argmin(avg_valid_lost))
    best_est = PolynomialFitting(min_mse_degree)
    best_est.fit(train_X.to_numpy(), train_y.to_numpy())
    best_est_loss = mean_square_error(test_y.to_numpy(), best_est.predict(test_X.to_numpy()))
    print(
        f"The degree with min MSE is: {min_mse_degree} and its error over the test set is: {np.round(best_est_loss, 2)}")


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
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y, test_X, test_y = X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    MSE_train_ridge = []
    MSE_test_ridge = []
    MSE_train_lasso = []
    MSE_test_lasso = []
    range_of_lambda = np.linspace(0.01, 2, n_evaluations)

    for lam in range_of_lambda:
        MSE_train_ridge_, MSE_test_ridge_ = cross_validate(RidgeRegression(lam), train_X, train_y, mean_square_error)
        MSE_train_lasso_, MSE_test_lasso_, = cross_validate(Lasso(alpha=lam), train_X, train_y, mean_square_error)
        MSE_train_ridge.append(MSE_train_ridge_)
        MSE_test_ridge.append(MSE_test_ridge_)
        MSE_train_lasso.append(MSE_train_lasso_)
        MSE_test_lasso.append(MSE_test_lasso_)

    go.Figure([
        go.Scatter(name='MSE train ridge', x=range_of_lambda, y=MSE_train_ridge, mode='markers', marker_color='blue'),
        go.Scatter(name='MSE test ridge', x=range_of_lambda, y=MSE_test_ridge, mode='markers', marker_color='orange'),
        go.Scatter(name='MSE train lasso', x=range_of_lambda, y=MSE_train_lasso, mode='markers', marker_color='green'),
        go.Scatter(name='MSE test lasso', x=range_of_lambda, y=MSE_test_lasso, mode='markers', marker_color='red')]) \
        .update_layout(title=r"$\text{(1) }\text{Scatter plot of training set, test set and true model}$",
                       xaxis_title=r"$\text{x}$",
                       yaxis_title=r"$\text{F(x)}$").show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge = range_of_lambda[np.argmin(MSE_test_ridge)]
    best_lasso = range_of_lambda[np.argmin(MSE_test_lasso)]
    print(
        f"""
         The best parameter for the ridge estimator is: {best_ridge}
         The best parameter for the lasso estimator is: {best_lasso}
        """)
    best_ridge_est = RidgeRegression(best_ridge).fit(train_X, train_y)
    best_lasso_est = Lasso(alpha=best_lasso).fit(train_X, train_y)
    linear_reg_est = LinearRegression().fit(train_X, train_y)
    print(
        f"""
         Ridge regression MSE: {np.round(mean_square_error(test_y, best_ridge_est.predict(test_X)), 2)}
         Lasso regression MSE: {np.round(mean_square_error(test_y, best_lasso_est.predict(test_X)), 2)}
         Linear regression MSE: {np.round(mean_square_error(test_y, linear_reg_est.predict(test_X)), 2)}
        """)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
