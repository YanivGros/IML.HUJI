from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        from sklearn.naive_bayes import GaussianNB
        self.clf = GaussianNB()
        self.clf.fit(X, y)
        n_features = X.shape[1]
        n_sample = X.shape[0]
        n_classes = np.unique(y).shape[0]
        self.classes_ = np.unique(y)
        self.mu_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))
        for i, cls in enumerate(self.classes_):
            samples_in_class = X[y == cls]
            self.mu_[i] = samples_in_class.mean(axis=0)
            self.vars_[i] = np.var(samples_in_class, ddof=1, axis=0)  # todo put ddof
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
        res = np.array([self.calc_gaussian_naive_bayes(x) for x in X])
        return res

    def calc_gaussian_naive_bayes(self, X: np.ndarray):
        prob_list = self.calc_likelihood(X)
        return self.classes_[np.argmax(prob_list)]

    def calc_likelihood(self, X):
        prob_list =[]
        for i in range(len(self.classes_)):
            a_k = ((X - self.mu_[i]) ** 2) / (2 * self.vars_[i])
            b_k = -np.log(np.sqrt(2 * np.pi * self.vars_[i]))
            c_k = np.log(self.pi_[i])
            res = (b_k - a_k).sum() + c_k
            prob_list.append(res)
        return prob_list

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
        from scipy.stats import norm
        # temp_1 =norm(self.mu_[0][0],self.vars_[0][0]).pdf(X[0][0])
        # temp_2 = norm(self.mu_[0][1],self.vars_[0][1]).pdf(X[0][1])
        # x_1_1 = temp_1 * temp_2
        res_list = np.array([self.calc_likelihood(x) for x in X])
        # a_k = ((X - self.mu_[i]) ** 2) / (2 * self.vars_[i])
        # b_k = -np.log(np.sqrt(2 * np.pi * self.vars_[i]))
        # c_k = np.log(self.pi_[i])
        # res = (b_k - a_k).sum() + c_k

        mu_rep = np.tile(self.mu_.T, X.shape[0]).T
        X_rep = np.repeat(X, self.mu_.shape[0], axis=0)
        var_rep = np.tile(self.vars_.T, X.shape[0]).T
        pi_rep = np.tile(self.pi_.reshape(-1, 1), X.shape[0]).T

        a_k = (X_rep - mu_rep) ** 2 / var_rep
        b_k  = -np.log(np.sqrt(2 * np.pi * var_rep))
        c_k  =np.log(pi_rep)
        res = (b_k - a_k).sum() + c_k

        pi_rep = np.tile(self.pi_.reshape(-1, 1), X.shape[0] ).T
        var_rep = np.tile(self.vars_.T, X.shape[0]).T
        mahalanobis = ((mu_rep - X_rep) / var_rep) ** 2
        res = np.exp(-.5 * mahalanobis) / np.sqrt(2 * np.pi * var_rep)
        return (res[:, 0] * res[:, 1]).reshape(X.shape[0], self.mu_.shape[0]) * pi_rep

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
