from __future__ import annotations
import numpy as np
from numpy import ndarray
from numpy.linalg import inv, det, slogdet
from numpy.random import multivariate_normal


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=True
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        self.var_ = X.var() if self.biased_ else X.var(ddof=1)
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """

        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        guas_func = lambda x: 1 / (np.sqrt(2 * np.pi * self.var_)) * np.exp(- (x - self.mu_) ** 2 / (2 * self.var_))
        ret = (list(map(guas_func, X)))
        return ret

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        # guas_func = lambda x: np.log(1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))
        const = -0.5 * len(X) * np.log(2 * np.pi * (sigma                                                                                                          ))
        sigma_divide = -1 / (2 * (sigma))
        sum_of_loss = np.sum(np.power(X - mu, 2))
        return  const + sigma_divide * sum_of_loss
        return -0.5 * len(X) * np.log(2 * np.pi * (sigma ** 2)) - (1 / 2 * (sigma ** 2)) * np.sum(np.power(X - mu, 2))

        #
        # return -0.5 * len(X) * np.log(2 * np.pi * (sigma ** 2)) - np.sum(np.power(X - mu, 2) / (2 * (sigma ** 2)))
        # return np.sum(
        #     np.log(np.divide(1, np.sqrt(2 * np.pi) * sigma) * np.exp(-np.divide(np.power(X - mu, 2), 2 * sigma ** 2))))
        # normalizer = -(np.divide(1, np.multiply(2, np.power(sigma, 2))))
        # adjust = -0.5 * len(X) * np.log(np.multiply(np.power(sigma, 2), np.pi) * 2)
        # the_sum = normalizer * sum(np.power(X - mu, 2))
        # return the_sum - adjust
        # return np.multiply(imUsingThis * normalizer, sum(np.power(X - mu, 2)))
        #
        # guas_func = lambda x: np.log(1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu) ** 2 / (2 * sigma ** 2)))
        # ret = (np.prod(list(map(guas_func, X))))
        # print(ret)
        # return ret
        #

class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """
    cov_: ndarray

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean(0)
        self.cov_ = np.cov(X, rowvar=False)

        # self.mu_ np.array() # todo should calc the mean for each col.
        # self.cov_ # todo should use the formala to clalculate the cov of the ith elment with the j element.
        self.fitted_ = True
        return self

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        # shape 0 col chape 1 row
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `pdf` function")
        fraction = np.power(2.0 * np.pi, len(self.cov_) / 2.0) * np.sqrt(np.linalg.det(self.cov_))
        multi_normal_pdf = lambda x: (1.0 / fraction) * np.exp(
            -0.5 * np.transpose(x - self.mu_) @ (np.linalg.inv(self.cov_)) @ (x - self.mu_))
        return map(multi_normal_pdf, X)
        # var = multivariate_normal(mean=[0, 0, 4, 0], cov=[[1, 0], [0, 1]])

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        m = len(X)
        d = len(mu)

        log_det = slogdet(cov)[1]
        d_log_pi = d * np.log(2.0 * np.pi)
        second = (m / 2.0) * (log_det + d_log_pi)
        # likelihood_function = lambda x: -0.5 * ((x - mu).T @ inv(cov) @ (x - mu))
        first = np.apply_along_axis(lambda x: -0.5 * ((x - mu).T @ inv(cov) @ (x - mu)), 1, X).sum()
        return first - second

        # likelihood_function = lambda x: -0.5 * ((x - mu).T @ inv(cov) @ (x - mu)) - second
        # print(list(map(likelihood_function, X)))
        # for x_ in X:
        #     print(likelihood_function(x_))

        #
