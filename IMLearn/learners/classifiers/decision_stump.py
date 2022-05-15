from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _calc_min_between_sign(self, X, y, j):

        sign_plus = self._find_threshold(X[:, j], y, 1)
        sign_minus = self._find_threshold(X[:, j], y, -1)
        if sign_plus[1] > sign_minus[1]:
            return *sign_minus, -1, j
        else:
            return *sign_plus, 1, j

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        key_ = min([self._calc_min_between_sign(X, y, j) for j in range(X.shape[1])], key=lambda x: x[1])
        self.threshold_ = key_[0]
        self.sign_ = key_[2]
        self.j_ = key_[3]

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return ((X[:, self.j_] < self.threshold_) * 2 - 1) * -self.sign_

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray, sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        # init
        sorted_index = np.argsort(values)
        labels_abs = np.abs(labels[sorted_index])
        labels_sign = np.sign(labels[sorted_index])
        sum_labels = np.sum(labels_abs)
        predicted_labels = np.ones(labels.shape[0]) * sign
        min_score = 1
        min_val = 0
        # calc score for each label and return min
        for i in range(labels.shape[0]):
            score = np.sum((labels_abs[predicted_labels != labels_sign])) / sum_labels
            if min_score > score:
                min_score = score
                min_val = values[sorted_index[i]]
            predicted_labels[i] *= -1
        return min_val, min_score

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return float(np.sum(np.abs(y[self.predict(X) != np.sign(y)])))


