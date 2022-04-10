from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    full_data = pd.read_csv(filename).drop_duplicates().dropna()

    omit_rows_index = full_data[
        (full_data['price'] < 1) | (full_data['zipcode'] < 1) | (full_data['bedrooms'] < full_data['bathrooms']) | (
                full_data['sqft_lot15'] < 1) | (full_data['bedrooms'] > 10) | (
                (full_data['yr_built'] > full_data['yr_renovated']) & (full_data['yr_renovated'] > 0))].index
    full_data.drop(omit_rows_index, inplace=True)

    date_to_days_from_starting_sale(full_data)

    start_reno = min_larger_than_zero(full_data['yr_renovated'])

    # replace the replace all the zero of the starting to renovate by the min value larger than zero
    full_data['yr_renovated'] = full_data['yr_renovated'].apply(lambda x: start_reno if x < start_reno else x)

    calc_zip_avg_cost(full_data)
    price = full_data['price']
    full_data.drop(columns=['id', 'price'], inplace=True)
    return full_data, price


def date_to_days_from_starting_sale(full_data):
    new_date = pd.to_datetime(full_data["date"])
    stat_buying_date = min(new_date)
    new_date -= stat_buying_date
    full_data['date'] = new_date.dt.days


def min_larger_than_zero(a):
    return a[a > 1].min()


def calc_zip_avg_cost(full_data):
    zip_to_cost = full_data.groupby('zipcode')['price'].mean()
    avg_zip_cost = zip_to_cost.mean()
    full_data['zipcode'] = full_data['zipcode'].map(zip_to_cost, avg_zip_cost)
    full_data.rename(columns={"zipcode": "avgzipcost"}, inplace=True)


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """

    y_std = np.std(y)

    def calc_pearson_col(X):
        col_std = np.std(X)
        cov_ = np.cov(X, y, rowvar=False)[0][1]
        return np.divide(cov_, y_std * col_std)

    for col in X:
        fig = px.scatter(X, x=col, y=y, labels={'y': 'price'},
                         title=f'Price as function of {col}. Pearson Correlation is: {calc_pearson_col(X[col])}')
        fig.show()
        fig.write_image(f"{output_path}\\{col}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_x, train_y, test_x, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    training_set = pd.concat([train_x, train_y], axis=1)
    n_train = len(training_set)
    estimator = LinearRegression()
    mean_list = []
    std_list = []
    for i in np.arange(0.10, 1.01, 0.01):
        list_res = []
        for j in range(10):
            sample = pd.DataFrame.sample(training_set, n=np.ceil(n_train * i).astype(int))
            estimator.fit(sample.loc[:, sample.columns != 'price'], sample['price'])
            res = estimator.loss(test_x, test_y)
            list_res.append(res)
        mean_list.append(np.mean(list_res))
        std_list.append(np.std(list_res))
    means = np.array(mean_list)
    std = np.array(std_list)
    go.Figure([go.Scatter(x=np.arange(0.10, 1.01, 0.01), y=means - 2 * std, fill=None, mode="lines",
                          line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=np.arange(0.10, 1.01, 0.01), y=means + 2 * std, fill='tonexty', mode="lines",
                          line=dict(color="lightgrey"),
                          showlegend=False),
               go.Scatter(x=np.arange(0.10, 1.01, 0.01), y=means, mode="markers+lines",
                          marker=dict(color="black", size=1), showlegend=False)],
              layout=go.Layout(
                  title=r"$\ Mean and Variance of Loss As Function Of Sample Size}$", )).show()
