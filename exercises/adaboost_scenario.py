import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
import plotly.express as px

from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, 0)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    clf = AdaBoost(DecisionStump, n_learners)
    clf.fit(train_X, train_y)
    res_training = []
    res_test = []
    range_ = np.arange(1, n_learners + 1)
    for i in range_:
        res_training.append(clf.partial_loss(train_X, train_y, i))
        res_test.append(clf.partial_loss(test_X, test_y, i))

    fig = go.Figure([
        go.Scatter(x=range_, y=res_test, name="test error", showlegend=True,
                   marker=dict(color="blue", opacity=.7),
                   line=dict(color="blue", width=1)),
        go.Scatter(x=range_, y=res_training, name="test error", showlegend=True,
                   marker=dict(color="black", opacity=.7),
                   line=dict(color="black", width=1))],
        layout=go.Layout(
            title=r"$\text{training and test errors as function of the number of fitted leraners }$",
            xaxis={"title": "x - Explanatory Variable"},
            yaxis={"title": "y - Response"}))
    fig.show()

    clf.predict(train_X)

    ada_loss = clf.loss(test_X, test_y)

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\textbf{{number of learners {t}}}$" for t in T],
                        horizontal_spacing=0.03, vertical_spacing=.06)

    for i, t_ in enumerate(T):
        fig.add_traces([decision_surface(lambda x: clf.partial_predict(x, t_), lims[0], lims[1], showscale=False),
                        go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=test_y, colorscale=[custom[0], custom[-1]]))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"$\textbf{{Decision boundary with different ensemble size}}$", margin=dict(t=100)) \
        .update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()
    print()
    print()
    print()
    # Question 3: Decision surface of best performing ensemble
    min_error_ind = np.argmin(res_test)

    fig = go.Figure(
        layout=go.Layout(
            title=rf"$\textbf{{Decision boundary of ensemble size :{min_error_ind + 1}, accuracy: {1 - clf.partial_loss(test_X, test_y, min_error_ind)}}}$"))
    fig.add_traces(
        [decision_surface(lambda x: clf.partial_predict(x, min_error_ind + 1), lims[0], lims[1], showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y))])
    fig.show()
    px.scatter(x=train_X[:, 0], y=train_X[:, 1], color=train_y,
               color_continuous_scale=[custom[0], custom[-1]],
               size=5 * clf.D_ / np.max(clf.D_),
               title="Training set plot with a point size proportional to it’s weight").show()
    # fig = go.Figure(
    #     layout=go.Layout(
    #         title=rf"$\textbf{{training set with a point size proportional to it’s weight}}$"))
    # fig.add_traces([go.Scatter(x=train_X[:, 0],
    #                            y=train_X[:, 1],
    #                            marker=dict(color=test_y[test_y == 1],
    #                                        colorscale=[custom[0], custom[-1]],
    #                                        size=5 * clf.D_ / np.max(clf.D_)))])

    # Question 4: Decision surface with weighted samples


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(.4)
    # f = np.arange(0, 20).reshape(2,10)
    # l = np.ones(10)
    # l[5:10] = -1
    # clf = DecisionStump()
    # clf.fit(f.T,l)
