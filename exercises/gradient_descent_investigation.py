import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from utils import *
import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights_list = []

    def callback(solver, weights, val, grad, t, eta, delta):
        values.append(np.array(val))
        weights_list.append(np.array(weights))

    return callback, values, weights_list


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    figl1 = go.Figure()
    figl2 = go.Figure()
    min_val_list_l1 = []
    min_val_list_l2 = []
    for eta in etas:
        call_, values_, weights_ = get_gd_state_recorder_callback()
        GradientDescent(FixedLR(eta), callback=call_).fit(L1(init), None, None)
        plot_descent_path(L1, np.array(weights_), f"of L1 with fixed learning rate: {eta}").show()
        figl1.add_trace(go.Scatter(x=np.arange(0, len(values_)), y=values_, mode='markers', name=f"Eta ={eta}"))
        min_val_list_l1.append(min(values_))
    for eta in etas:
        call_, values_, weights_ = get_gd_state_recorder_callback()
        GradientDescent(FixedLR(eta), callback=call_).fit(L2(init), None, None)
        plot_descent_path(L2, np.array(weights_), f"of L2 with fixed learning rate: {eta}").show()
        figl2.add_trace(go.Scatter(x=np.arange(0, len(values_)), y=values_, mode='markers', name=f"Eta = {eta}"))
        min_val_list_l2.append(min(values_))

    figl1.update_layout(
        title="Norm as a function of the GD iteration of L1 objective function with fixed learning rate",
        xaxis_title=r"$\text{Iteration}$",
        yaxis_title=r"$\text{Norm   }$").show()
    figl2.update_layout(
        title="Norm as a function of GD iteration of L2 objective function with fixed learning rate",
        xaxis_title=r"$\text{Iteration}$",
        yaxis_title=r"$\text{Norm}$").show()
    print(f"min_value of l1 is {min(min_val_list_l1)}")
    print(f"min_value of l2 is {min(min_val_list_l2)}")


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate

    fig = go.Figure()
    min_val_list = []
    for gama in gammas:
        call_, values_, weights_ = get_gd_state_recorder_callback()
        GradientDescent(ExponentialLR(eta, gama), callback=call_).fit(L1(init), None, None)
        # Plot algorithm's convergence for the different values of gamma
        fig.add_trace(go.Scatter(x=np.arange(0, len(values_)), y=values_, mode='markers', name=f"gamma = {gama}"))
        min_val_list.append(min(values_))
        # Plot descent path for gamma=0.95
        if gama == .95:
            plot_descent_path(L1, np.array(weights_), f"of L1 with exponentially decaying learning rate: {gama}").show()
    fig.update_layout(
        title="Norm as a function of GD iteration of L1 objective function with exponentially decaying learning rate",
        xaxis_title=r"$\text{Iteration}$",
        yaxis_title=r"$\text{Norm}$").show()
    print(f"min_val of exponential decay rates is {min(min_val_list)} ")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()
    X_train, y_train, X_test, y_test = X_train.to_numpy(), y_train.to_numpy(), X_test.to_numpy(), y_test.to_numpy(),
    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic_reg = LogisticRegression().fit(X=X_train, y=y_train)
    c = [custom[0], custom[-1]]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(y_train, logistic_reg.predict_proba(X_train))

    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                         xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                         yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).show()
    best_alpha = thresholds[np.argmax(tpr - fpr)]
    print(f"best alpha is {best_alpha}")
    logistic_reg.alpha_ = best_alpha
    print(f"model test error is {logistic_reg.loss(X_test, y_test)}")

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    from IMLearn.metrics.loss_functions import misclassification_error
    lam_list = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)
    score_list_l1 = []
    score_list_l2 = []
    for lam in lam_list:
        l1_rgz_lgt_reg = LogisticRegression(penalty="l1", lam=lam)
        score_list_l1.append(cross_validate(l1_rgz_lgt_reg, X_train, y_train, misclassification_error)[1])
    best_lam = lam_list[int(np.argmin(score_list_l1))]
    best_l1_lgt_reg = LogisticRegression(penalty="l1", lam=best_lam)
    best_l1_lgt_reg.fit(X_train, y_train)
    print(f"l1 regularization, Best lamda is {best_lam} and it's test error is {best_l1_lgt_reg.loss(X_test, y_test)}")
    for lam in lam_list:
        l2_rgz_lgt_reg = LogisticRegression(penalty="l2", lam=lam)
        score_list_l2.append(cross_validate(l2_rgz_lgt_reg, X_train, y_train, misclassification_error)[1])
    best_lam = lam_list[int(np.argmin(score_list_l2))]
    best_l2_lgt_reg = LogisticRegression(penalty="l2", lam=best_lam)
    best_l2_lgt_reg.fit(X_train, y_train)
    print(f"l2 regularization, Best lamda is {best_lam} and it's test error is {best_l2_lgt_reg.loss(X_test, y_test)}")


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()
# best alpha is 0.32477367537820134
# model test error is 0.33695652173913043
# l1 regularization, Best lamda is 0.001 and it's test error is 0.31521739130434784
# l1 regularization, Best lamda is 0.001 and it's test error is 0.6847826086956522
# l1 regularization, Best lamda is 0.1 and it's test error is 0.31521739130434784
# l2 regularization, Best lamda is 0.001 and it's test error is 0.31521739130434784

