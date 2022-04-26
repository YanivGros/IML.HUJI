import numpy as np
from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        losses = []

        def writing_loss_callback(fit_: Perceptron, x_: np.ndarray, y_: int):
            losses.append(fit_.loss(X, y))

        p = Perceptron(callback=writing_loss_callback)
        p.fit(X, y)

        # Plot figure of loss as function of fitting iteration
        go.Figure(go.Scatter(x=list(range(len(losses))), y=losses, mode="markers + lines", marker=dict(color="black")),
                  layout=dict(template="simple_white",
                              title=f"Perceptron Algorithm Loss Values as Function of Training Iterations over {n}"
                                    f" Data Set",
                              xaxis_title="Training Iterations",
                              yaxis_title="Loss Values")).show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")

        # Fit models and predict over training set
        LDA_classifier = LDA()
        guas = GaussianNaiveBayes()
        guas.fit(X, y)
        res = guas.likelihood(X)
        LDA_classifier(X)
        np.array([0, 1])
        S_1 = [0, 1, 2, 3, 4, 5, 6, 7]
        S_2 = [0, 0, 1, 1, 1, 1, 2, 2]
        guas.fit(np.array(S_1).reshape(-1, 1), np.array(S_2).reshape(-1, 1))

        S_1 = np.array(([1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]))
        S_2 = np.array([0, 0, 1, 1, 1, 1])
        S_ = [([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)]
        S = np.array([([1, 1], 0), ([1, 2], 0), ([2, 3], 1), ([2, 4], 1), ([3, 3], 1), ([3, 4], 1)])
        np.array(S)
        temp = np.array(S[:, 0])
        Y_ = np.array(S[:, 1])

        LDA_classifier.fit(S[:, 0], S[:, 1])
        LDA_classifier.fit(X, y)
        LDA_classifier.likelihood(X)
        LDA_classifier.loss(X, y)
        guas.fit(X, y)
        guas.predict(X)
        guas.likelihood(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from IMLearn.metrics import accuracy
        y_pred_gaus = guas.predict(X)
        y_pred_lda = LDA_classifier.predict(X)
        symbols = np.array(["circle", "diamond", "triangle-up"])
        fig = make_subplots(1, 2, subplot_titles=[f"GNB predicted class, Accuracy: {accuracy(y, y_pred_gaus)}",
                                                  f"LDA predicted class, Accuracy: {accuracy(y, y_pred_lda)}"])
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                   marker=dict(color=y_pred_gaus, symbol=symbols[y],
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=1)
        fig.add_traces([go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                   marker=dict(color=y_pred_lda, symbol=symbols[y],
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=2)

        # Add `X` dots specifying fitted Gaussians' means

        fig.add_traces([go.Scatter(x=guas.mu_[:, 0], y=guas.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x",
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=1)

        fig.add_traces([go.Scatter(x=LDA_classifier.mu_[:, 0], y=LDA_classifier.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x",
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=2)

        # Add ellipses depicting the covariances of the fitted Gaussians

        for i in range(guas.mu_.shape[0]):
            fig.add_traces(get_ellipse(guas.mu_[i], guas.vars_[i] * np.identity(2)), rows=1, cols=1)
        for i in range(LDA_classifier.mu_.shape[0]):
            fig.add_traces(get_ellipse(LDA_classifier.mu_[i], LDA_classifier.cov_), rows=1, cols=2)
        fig.update_layout(title=f"Comparing LDA and GNB, DataSet: {f}", margin=dict(t=100)).show()

        # raise NotImplementedError()
        #

        # raise NotImplementedError()


# todo clean code
# todo make sure everthing work


if __name__ == '__main__':
    np.random.seed(0)
    # foo = np.ones((3, 4))
    # foo = np.array([[1, 2, 3, 4], [2, 2, 2, 3], [5, 1, 7, 3], [2, 4, 9, 1]])
    # foo_2 = np.zeros((3, 1))
    # l = LDA()
    # l.fit(X=foo, y=np.array([1, 1, 2, 2]))
    # exit(0)

    # run_perceptron()

    compare_gaussian_classifiers()
