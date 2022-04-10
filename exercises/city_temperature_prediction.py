import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    full_data = pd.read_csv(filename, parse_dates=["Date"]).drop_duplicates().dropna()
    omit_rows_index = full_data[(full_data['Temp'] < -30)].index
    full_data.drop(omit_rows_index, inplace=True)
    full_data['DayOfYear'] = full_data['Date'].dt.dayofyear
    return full_data.loc[:, full_data.columns != 'Temp'], full_data['Temp']


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset

    X, y = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country

    israel_ind = X['Country'] == 'Israel'
    X_isr, y_isr = X[israel_ind], y[israel_ind]
    disc_year = X_isr['Year'].astype(str)

    px.scatter(X_isr, x='DayOfYear', y=y_isr, labels={'y': 'Temperature', 'DayOfYear': 'Day of year'},
               color=disc_year,
               title="Average daily temperature as a function of Day of year .").show()

    israel_data = pd.concat([X_isr, y_isr], axis=1)
    israel_std_by_month = israel_data.groupby('Month').std().reset_index()
    px.bar(israel_std_by_month, x='Month', y='Temp', labels={'Temp': 'Temperature STD'}).show()

    # Question 3 - Exploring differences between countries

    res = pd.concat([X, y], axis=1).groupby(['Country', 'Month'])
    mean = res.mean().reset_index()
    var = res.std().reset_index()
    mean['Temp Variance'] = var['Temp']
    mean['Temp Mean'] = mean['Temp']
    graph_data = mean
    px.line(graph_data, x='Month', y='Temp Mean', color='Country', error_y='Temp Variance',
            title="Average monthly temperature as function of month in different countries ").show()

    # Question 4 - Fitting model for different values of `k`

    train_x, train_y, test_x, test_y = split_train_test(X_isr, y_isr)
    train_x = train_x["DayOfYear"]
    test_x = test_x["DayOfYear"]
    loss_list = []
    for k in range(1, 11):
        p_fit = PolynomialFitting(k)
        p_fit.fit(train_x.to_numpy(), train_y.to_numpy())
        loss = np.round((p_fit.loss(test_x.to_numpy(), test_y.to_numpy())), decimals=2)
        loss_list.append(loss)
        print(f"Degree is: {k}, loss is {loss}")
    px.bar(x=np.arange(1, 11), y=loss_list, labels={'x': 'Degree', 'y': 'Loss'},
           title="Loss of the model as function of the degree of the polynomial fitting for Israel").show()

    # Question 5 - Evaluating fitted model on different countries

    p_fit_israel = PolynomialFitting(5)
    p_fit_israel.fit(X_isr['DayOfYear'].to_numpy(), y_isr)
    all_count = pd.unique(X['Country'])
    loss_per_country = {}
    for country in all_count:
        if country == 'Israel':
            continue
        X_country, y_Country, = X[X['Country'] == country], y[X['Country'] == country]
        country_loss = p_fit_israel.loss(X_country['DayOfYear'].to_numpy(), y_Country)
        loss_per_country[country] = country_loss
    country_loss = pd.DataFrame.from_dict(loss_per_country, orient='index')
    px.bar(country_loss, labels={'index': 'Country', 'value': 'Loss'},
           title="Loss of model fitted for Israel as function of country").show()
