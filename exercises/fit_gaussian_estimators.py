import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian

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
    eug = UnivariateGaussian(False)
    for m in ms:
        estimated_mean.append(abs(eug.fit(sample_size_1000[0:m]).mu_ - mu))
    go.Figure([go.Scatter(x=ms, y=estimated_mean, mode='markers+lines', name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{ Absolute distance between the estimated- and true value of the expectation as a "
                        r"function of sample size}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title="r$|\hat\mu-\mu|$")).show()

    # # Question 3 - Plotting Empirical PDF of fitted model
    sample_size_1000.sort()
    go.Figure([go.Scatter(x=sample_size_1000, y=estimator_ug.pdf(sample_size_1000), mode="markers+lines",
                          name="PDF")],
              layout=go.Layout(title="PDF function under the fitted model",
                               xaxis_title="Samples",
                               yaxis_title="Density", )).show()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mean = np.array([0, 0, 4, 0])

    cov = np.array([[1, 0.2, 0, 0.5],
                    [0.2, 2, 0, 0],
                    [0, 0, 1, 0],
                    [0.5, 0, 0, 1]])
    size = 1000
    samples_multivariate_normal = np.random.multivariate_normal(mean, cov, size)
    emg = MultivariateGaussian()
    emg.fit(samples_multivariate_normal)

    print(emg.mu_)
    print(emg.cov_)

    # Question 5 - Likelihood evaluation
    f1 = np.linspace(-10, 10, 200)
    f3 = np.linspace(-10, 10, 200)

    temp_likelihood_matrix = []
    for i in f1:
        cur_ = []
        for j in f3:
            cur_.append(emg.log_likelihood([i, 0, j, 0], cov, samples_multivariate_normal))
        temp_likelihood_matrix.append(cur_)
    likelihood_matrix = np.array(temp_likelihood_matrix)
    fig = go.Figure(
        layout=go.Layout(title="Log-likelihood of a model as a function of features 1 and 3 of the variance",
                         xaxis_title="f1",
                         yaxis_title="f3", ),
        data=go.Heatmap(x=f1, y=f3, z=likelihood_matrix))
    fig.show()


    # Question 6 - Maximum likelihood
    ind = np.unravel_index(np.argmax(likelihood_matrix), likelihood_matrix.shape)
    print(round(f1[ind[0]], ndigits=3), round(f3[ind[1]], ndigits=3))

if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
