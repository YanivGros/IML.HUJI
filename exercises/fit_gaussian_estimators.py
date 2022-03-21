import numpy
from scipy.stats import norm

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu, sigma = 10, 1
    sample_size_1000 = np.random.normal(mu, sigma, 1000)
    estimator_ug = UnivariateGaussian(False)
    estimator_ug.fit(sample_size_1000)
    print(f"({estimator_ug.mu_},{estimator_ug.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    ms = np.linspace(10, 1000, 100).astype(int)
    estimated_mean = []
    un = UnivariateGaussian(False)
    for m in ms:
        estimated_mean.append(abs(un.fit(sample_size_1000[0:m]).mu_ - mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{ Absolute distance between the estimated- and true value of the expectation as a "
                        r"function of sample size}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu-\mu|$")).show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    sample_size_1000.sort()
    go.Figure([go.Scatter(x=sample_size_1000, y=estimator_ug.pdf(sample_size_1000), mode="markers",
                          name="PDF")],
              layout=go.Layout(title="PDF function under the fitted model",
                               xaxis_title="Samples",
                               yaxis_title="Density", )).show()
    arr = np.array( # todo delete
        [1, 5, 2, 3, 8, -4, -2, 5, 1, 10, -10, 4, 5, 2, 7, 1, 1, 3, 2, -1, -3, 1, -4, 1, 2, 1, -4, -4, 1, 3, 2, 6, -6,8, 3, -6, 4, 1, -2, 3, 1, 4, 1, 4, -2, 3, -1, 0, 3, 5, 0, -2])
    print(un.log_likelihood(1,1,arr))
    print(un.log_likelihood(10,1,arr))



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])

    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    size = 1000
    samples_multivariate_normal = numpy.random.multivariate_normal(mean, cov, size)
    emg = MultivariateGaussian()
    emg.fit(samples_multivariate_normal)

    print(emg.mu_)
    print(emg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)
    print(emg.log_likelihood([0.5263157894736832, 0, 3.6842105263157894, 0], cov, samples_multivariate_normal))

    def like_fun(x: float, y: float):
        return np.add(emg.log_likelihood([x, 0, y, 0], cov, samples_multivariate_normal), 0)

    temp_likelihood_matrix = []
    for i in f1:
        cur_ = []
        for j in f3:
            cur_.append(like_fun(i, j))
        temp_likelihood_matrix.append(cur_)
    likelihood_matrix = np.array(temp_likelihood_matrix)
    fig = go.Figure(data=go.Heatmap(x=f1, y=f3, z=likelihood_matrix))
    fig.show()

    # Question 6 - Maximum likelihood
    # raise NotImplementedError()
    print(np.argmax(likelihood_matrix))
    print(emg.log_likelihood([0.5263157894736832, 0, 3.6842105263157894, 0], cov, samples_multivariate_normal))
    ind = np.unravel_index(np.argmax(likelihood_matrix), likelihood_matrix.shape)
    print(ind)
    print(likelihood_matrix[ind])
    print(f1[ind[0]], f3[ind[1]])

    # print(f1[5],f3[6])


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
