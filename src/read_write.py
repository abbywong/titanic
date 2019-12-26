
"""
These functions read the input, and write the output.
"""

from pandas import read_csv


def read_and_tune_csv_data(fname):
    data = read_csv("data/{}".format(fname), dtype=dict(
        Sex='category',
        Cabin='category',
        Embarked='category',
    ))
    return data


def read_train_data():
    return read_and_tune_csv_data("train.csv")


def read_test_data():
    return read_and_tune_csv_data("train.csv")

