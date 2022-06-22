from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    validations = np.array_split(np.argwhere(y), cv)
    train_loss_list = []
    valid_lost_list = []
    for v in validations:
        v = v.flatten()
        part_train_X, part_train_y = np.delete(X, v, axis=0), np.delete(y, v, axis=0)
        validations_X, validations_y = X[v], y[v]
        model = estimator.fit(part_train_X, part_train_y)
        train_loss_list.append(scoring(part_train_y, model.predict(part_train_X)))
        valid_lost_list.append(scoring(validations_y, model.predict(validations_X)))
    return np.mean(train_loss_list), np.mean(valid_lost_list)
