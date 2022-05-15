from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from itertools import product
from ...metrics.loss_functions import misclassification_error


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
        # min_all = 1
        # for j in range(X.shape[1]):
        #     min_ = self._calc_min_between_sign(X, y, j)
        #     if min_[1] < min_all[1]:
        #         min
        #
        # from sklearn.tree import DecisionTreeClassifier
        # self.real_clf = DecisionTreeClassifier(max_depth=1,max_leaf_nodes=2)
        # self.real_clf.fit(X, np.sign(y))

        # key_ = min([self._calc_min_between_sign(X, y, j) for j in range(X.shape[1])], key=lambda x: x[1])

        # all_l = [self._calc_min_between_sign(X, y, j) for j in range(X.shape[1])]
        # key = min(all_l, key=lambda x: x[1])
        key_ = min([self._calc_min_between_sign(X, y, j) for j in range(X.shape[1])],key=lambda x: x[1])
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
        sotred_index = np.argsort(values)
        sorted_vals = np.sort(values)
        sorted_labels = np.take(labels, sotred_index)
        sign_labels = np.sign(sorted_labels)
        min_thr = np.inf
        min_thr_err = 1
        temp_labels_by_thr = np.ones(values.shape[0]) * sign

        for i in range(sorted_vals.shape[0]):
            temp_err_by_thr = np.sum(np.where(temp_labels_by_thr != sign_labels,
                                              np.abs(sorted_labels), 0)) / len(sorted_vals)
            if temp_err_by_thr < min_thr_err:
                min_thr_err = temp_err_by_thr
                min_thr = sorted_vals[i]
            temp_labels_by_thr[i] = -sign
        return min_thr, min_thr_err
        # score_list = []
        # labels_abs = np.abs(labels)
        # sum_labels = np.sum(labels_abs)
        # labels_sign = np.sign(labels)
        # res = values
        #
        # min_score = 2  # score <= 1
        # min_val = 0
        # # calc score for each values as threshold
        # for value in values:
        #     values_sign = np.sign(values - value)
        #     # values_sign[values_sign == 0] = 1
        #     values_sign *= sign
        #     score = labels_abs[labels_sign != values_sign].sum() / sum_labels
        #     if min_score > score:
        #         min_score = score
        #         min_val = value
        # return min_val, min_score

        # score_list.append(labels_abs[labels_sign != values_sign].sum() / sum_labels)
        # min_ind = np.argmin(score_list)
        # return values[min_ind], score_list[min_ind]

        # res = []
        # for value in values:
        #     b = np.sign(values - value)
        #     b[b == 0] = 1
        #     b *= sign
        #     res.append(misclassification_error(b, labels))
        # min_ind = np.argmin(res)
        # return values[min_ind], res[min_ind]

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


if __name__ == '__main__':
    f = np.arange(0, 10)
    l = np.ones_like(10)
    l[np.arange(0, 10, 2)] = -1
