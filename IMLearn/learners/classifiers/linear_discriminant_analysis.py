from typing import NoReturn

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        # self.lda_real = LinearDiscriminantAnalysis(store_covariance=True)
        # self.lda_real.fit(X, y)

        n_features = X.shape[1]
        n_sample = X.shape[0]
        n_classes = np.unique(y).shape[0]
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((n_classes, n_features))
        self.cov_ = np.zeros((n_features, n_features))
        for i, cls in enumerate(self.classes_):
            temp = X[y == cls]
            self.mu_[i] = temp.mean(axis=0)
            temp_2 = temp - self.mu_[i]
            self.cov_ += (temp_2.T @ temp_2)
        self.cov_ /= (n_sample - n_classes)

        self._cov_inv = inv(self.cov_)
        self.pi_ = np.unique(y, return_counts=True)[1] / n_sample

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        res = np.array([self.calc_bayes_optimal_classifier(x) for x in X])
        return res

    def calc_bayes_optimal_classifier(self, X: np.ndarray):
        prob_list = []
        for i in range(len(self.classes_)):
            a_k = self._cov_inv @ self.mu_[i].T
            b_k = np.log(self.pi_[i]) - 0.5 * self.mu_[i] @ self._cov_inv @ self.mu_[i].T
            prob_list.append(a_k.T @ X + b_k)
        return self.classes_[np.argmax(prob_list)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        mu_rep = np.tile(self.mu_.T, X.shape[0]).T
        X_rep = np.repeat(X, self.mu_.shape[0], axis=0)
        mu_X = mu_rep - X_rep

        # from scipy.stats import multivariate_normal
        # mul = multivariate_normal.pdf(X, mean=self.mu_[0], cov=self.cov_)
        # temp = mu_X @ self._cov_inv @ mu_X.T
        mahalanobis = np.einsum("bi,ij,bj->b", mu_X, self._cov_inv, mu_X)
        mahalanobis = mahalanobis.reshape(X.shape[0], self.mu_.shape[0])
        res = np.exp(-.5 * mahalanobis) / np.sqrt((2 * np.pi) ** X.shape[1] * det(self.cov_))
        pi_rep = np.tile(self.pi_.reshape(-1, 1), X.shape[0]).T
        return res * pi_rep

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
        from ...metrics import misclassification_error
        return misclassification_error(self.predict(X), y)
