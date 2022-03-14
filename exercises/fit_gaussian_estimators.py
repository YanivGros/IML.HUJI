import numpy

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    sample_size_1000 = numpy.random.normal(1, 10, 1000)
    estimator = UnivariateGaussian(False)
    estimator.fit(sample_size_1000)
    print(f"({estimator.mu_},{estimator.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    for i in range(10, 1001, 10):
        current_estimator = UnivariateGaussian(True)
        current_estimator.fit(sample_size_1000[0:i])


    raise NotImplementedError()
    np.g

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
