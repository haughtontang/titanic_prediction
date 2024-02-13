from pathlib import Path

import numpy as np
import pandas as pd


def transform_data(data_df: pd.DataFrame, cols_to_normalise: list):
    for col in cols_to_normalise:
        data_df = mean_normalization(data_df=data_df, col_to_normalise=col)
    data_df = convert_sex_to_number(data_df=data_df)
    data_df = convert_location_to_number(data_df=data_df)
    data_df = add_family_size(data_df=data_df)
    data_df.index = data_df["PassengerId"]
    return data_df

def add_family_size(data_df: pd.DataFrame):
    data_df["family_size"] = data_df["SibSp"] + data_df["Parch"]
    return data_df


def convert_sex_to_number(data_df: pd.DataFrame):
    data_df["Sex"] = data_df["Sex"].replace("female", 0)
    data_df["Sex"] = data_df["Sex"].replace("male", 1)
    return data_df


def convert_location_to_number(data_df: pd.DataFrame):
    data_df["Embarked"] = data_df["Embarked"].replace("S", 1)
    data_df["Embarked"] = data_df["Embarked"].replace("C", 2)
    data_df["Embarked"] = data_df["Embarked"].replace("Q", 3)
    return data_df


def mean_normalization(data_df: pd.DataFrame, col_to_normalise: str):
    col_data = data_df[col_to_normalise]
    col_mean = col_data.mean()
    divider = max(col_data) - min(col_data)
    data_df[f"{col_to_normalise}_norm"] = (col_data - col_mean) / divider
    return data_df


def get_median_for_entire_data_set(test_df, train_df: pd.DataFrame, column_name: str):
    all_data_df = pd.concat([test_df, train_df])
    median = np.median(all_data_df[column_name].dropna().to_list())
    train_df[column_name] = train_df[column_name].replace(np.nan, median)
    test_df[column_name] = test_df[column_name].replace(np.nan, median)
    return train_df, test_df
