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
        lda_clf = LDA()
        gnb_clf = GaussianNaiveBayes()
        lda_clf.fit(X, y)

        gnb_clf.fit(X, y)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots

        from IMLearn.metrics import accuracy
        y_pred_gaus = gnb_clf.predict(X)
        y_pred_lda = lda_clf.predict(X)
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

        fig.add_traces([go.Scatter(x=gnb_clf.mu_[:, 0], y=gnb_clf.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x",
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=1)

        fig.add_traces([go.Scatter(x=lda_clf.mu_[:, 0], y=lda_clf.mu_[:, 1], mode="markers",
                                   marker=dict(color="black", symbol="x",
                                               line=dict(color="black", width=1),
                                               size=10), showlegend=False)], rows=1, cols=2)

        # Add ellipses depicting the covariances of the fitted Gaussians

        for i in range(gnb_clf.mu_.shape[0]):
            fig.add_traces(get_ellipse(gnb_clf.mu_[i], gnb_clf.vars_[i] * np.identity(2)), rows=1, cols=1)
        for i in range(lda_clf.mu_.shape[0]):
            fig.add_traces(get_ellipse(lda_clf.mu_[i], lda_clf.cov_), rows=1, cols=2)
        fig.update_layout(title=f"Comparing LDA and GNB, DataSet: {f}", margin=dict(t=100)).show()


if __name__ == '__main__':
    np.random.seed(0)

    run_perceptron()

    compare_gaussian_classifiers()
