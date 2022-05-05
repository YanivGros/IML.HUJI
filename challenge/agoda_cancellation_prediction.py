import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import missingno as msno
from sklearn import preprocessing
from sklearn.base import BaseEstimator
from sklearn.ensemble import AdaBoostClassifier

from IMLearn.utils import split_train_test
from agoda_cancellation_estimator import AgodaCancellationEstimator


def calc_canceling_fund(estimated_vacation_time,
                        cancelling_policy_code,
                        original_selling_amount, normalize=True):
    policy_options = cancelling_policy_code.split("_")
    cost_sum = 0
    for option in policy_options:
        if "D" in option:
            if "P" in option:
                charge = int(option[option.find("D") + 1:option.find("P")])
                charge /= 100
                cost_sum += original_selling_amount * charge
            if "N" in option:
                charge = int(option[option.find("D") + 1:option.find("N")])
                charge /= estimated_vacation_time
                cost_sum += original_selling_amount * charge
        elif "P" in option:
            charge = int(option[option.find("D") + 1:option.find("P")])
            charge /= 100
            cost_sum += original_selling_amount * charge
    if normalize:
        return ((cost_sum / len(policy_options)) / original_selling_amount) * 100
    return cost_sum / len(policy_options)


def add_hour_of_day(full_data: DataFrame):
    full_data["hour"] = pd.to_datetime(full_data["booking_datetime"]).dt.hour


def add_month_of_booking(full_data: DataFrame):
    full_data["month_of_booking"] = pd.to_datetime(full_data["booking_datetime"]).dt.month


def add_month_of_cheking(full_data):
    full_data["month_of_checkin"] = pd.to_datetime(full_data["checkin_date"]).dt.month


def data_conversion(full_data: DataFrame):
    country_to_hashcode(full_data)
    add_hour_of_day(full_data)
    add_month_of_booking(full_data)
    add_month_of_cheking(full_data)
    calc_checking_checkout(full_data)
    calc_booking_checking(full_data)
    enc = preprocessing.OrdinalEncoder()
    # enc.fit(full_data.)

    # full_data['month'].value_counts().plot.bar()
    # plt.show()
    accommodation_to_hashcode(full_data)
    payment_type_to_hashcode(full_data)
    if "cancellation_datetime" in full_data:
        calc_booking_canceling(full_data)
    calculate_canceling_fund_present(full_data)
    country_code(full_data)


def payment_type_to_hashcode(full_data):
    payment_type = set(full_data["original_payment_type"])
    payment_type_dict = {k: v for v, k in enumerate(payment_type)}
    full_data.replace({"original_payment_type": payment_type_dict},
                      inplace=True)
    bool_dict = {True: 0, False: 1}
    full_data.replace({"is_first_booking": bool_dict}, inplace=True)


def accommodation_to_hashcode(full_data):
    accommodation_type = set(full_data["accommadation_type_name"])
    accommodation_type_dict = {k: v for v, k in enumerate(accommodation_type)}
    full_data.replace({"accommadation_type_name": accommodation_type_dict},
                      inplace=True)


def country_to_hashcode(full_data):
    countries_code = set(full_data["origin_country_code"]).union(
        set(full_data["hotel_country_code"]))

    countries_code_dict = {k: v for v, k in enumerate(countries_code)}
    full_data.replace({"origin_country_code": countries_code_dict},
                      inplace=True)
    full_data.replace({"hotel_country_code": countries_code_dict},
                      inplace=True)


def calculate_canceling_fund_present(full_data):
    temp = full_data[["cancellation_policy_code", "original_selling_amount", "estimated_stay_time"]]
    res = temp.apply(lambda x: calc_canceling_fund(x['estimated_stay_time'], x['cancellation_policy_code'],
                                                   x['original_selling_amount']), axis=1)
    res_ = temp.apply(lambda x: calc_canceling_fund(x['estimated_stay_time'], x['cancellation_policy_code'],
                                                    x['original_selling_amount'], False), axis=1)
    full_data["avg_cancelling_fund_percent"] = res
    full_data["total_cancelling_fund"] = res_


def calc_booking_checking(full_data):
    checking_date = pd.to_datetime(full_data["checkin_date"])
    booking_date = pd.to_datetime(full_data["booking_datetime"])
    full_data["time_from_booking_to_check_in"] = (checking_date - booking_date).dt.days + 1


def calc_checking_checkout(full_data):
    checking_date = pd.to_datetime(full_data["checkin_date"])
    checkout_date = pd.to_datetime(full_data["checkout_date"])
    full_data["estimated_stay_time"] = (checkout_date - checking_date).dt.days + 1


def country_code(full_data: DataFrame):
    full_data["is_abroad"] = full_data["hotel_country_code"] == full_data["origin_country_code"]
    full_data["is_abroad"].astype(int)


def calc_booking_canceling(full_data):
    booking_date = pd.to_datetime(full_data["booking_datetime"])
    cancel_date = pd.to_datetime(full_data["cancellation_datetime"])
    full_data["cancelling_days_from_booking"] = (cancel_date - booking_date).dt.days + 1


def charge_option_to_hashcode(full_data):
    charge_option = set(full_data["charge_option"])
    num_charge_option = {value: key for key, value in
                         enumerate(charge_option)}
    full_data.replace({"charge_option": num_charge_option}, inplace=True)


def preprocess_data(df: pd.DataFrame):
    data_proc = []
    if_cancel_label = (~data_proc['cancellation_datetime'].isna()).astype(int)
    data_proc['is_canceled'] = if_cancel_label

    correlation = data_proc.corr()['is_canceled'].abs().sort_values(ascending=False)
    print(correlation)
    data_conversion(data_proc)
    features = pd.concat([data_proc, pd.get_dummies(full_data["charge_option"])], axis=1)
    if "cancellation_datetime" in full_data.columns:
        labels = full_data["cancelling_days_from_booking"]
    else:
        labels = []
    return features, labels


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    return pd.read_csv(filename)


def evaluate_data(dt: pd.DataFrame):
    msno.bar(dt)
    plt.show()
    msno.heatmap(dt)
    plt.show()


def drop_data(full_data: pd.DataFrame):
    full_data = full_data.drop_duplicates(inplace=False)
    # evaluate_data(full_data)
    full_data.drop(['hotel_brand_code',
                    'hotel_chain_code',
                    'request_largebed',
                    'request_earlycheckin',
                    'request_airport',
                    'request_highfloor',
                    'request_latecheckin',
                    'request_nonesmoke',
                    'request_twinbeds'],
                   axis=1,
                   inplace=True)
    # more than 40 present is missing (not including the canceling date which 73.91193167288907% nan)
    # features = full_data.drop('cancellation_datetime', axis=1, inplace=False)
    # full_data = full_data[~features.isna().any(1)].
    full_data.dropna(subset=full_data.columns.drop("cancellation_datetime"), inplace=True)
    return full_data


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray, filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    prediction = estimator.predict(X)
    res = []
    for i in range(len(X)):
        cancel_estimated = pd.to_datetime(X[i, 0]) + pd.to_timedelta(prediction[i], unit="days")
        if pd.to_datetime("2018-12-07") <= cancel_estimated <= pd.to_datetime("2018-12-13"):
            res.append(1)
        else:
            res.append(0)
    pd.DataFrame(res, columns=["predicted_values"]).to_csv(filename, index=False)
    return


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df = load_data("../datasets/agoda_cancellation_train.csv")
    df = drop_data(df)

    df = preprocess_data(df)

    cancellation_labels = []
    # Fit model over data
    df.head()

    train_X, train_y, temp_X, temp_y = split_train_test(df, cancellation_labels)  # for train-set
    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    temp_X = temp_X.to_numpy()
    temp_y = temp_y.to_numpy()
    # evaluate_different_estimator(train_X,train_y,temp_X,temp_y)
    # exit()
    estimator = AgodaCancellationEstimator().fit(train_X, train_y)
    print(estimator.loss(np.array(temp_X), np.array(temp_y)))

    # Store model predictions over test set
    df_test, cancellation_labels_test = load_data("test_set_week_3.csv")
    df_test = df_test.reindex(columns=df.columns, fill_value=0)
    test_X, test_y, temp_X2, temp_y2 = split_train_test(df_test, cancellation_labels_test, train_proportion=1)
    test_X = test_X.fillna(0).to_numpy()
    estimator.loss()
    evaluate_and_export(estimator, test_X, "208385633_315997874_206948911_test_2.csv")  # for test-set
