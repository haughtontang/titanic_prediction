from pathlib import Path
import pandas as pd


def load_and_transform_data(data_df: pd.DataFrame, cols_to_normalise: list):
    for col in cols_to_normalise:
        data_df = mean_normalization(data_df=data_df, col_to_normalise=col)
    data_df = convert_sex_to_number(data_df=data_df)
    data_df = convert_location_to_number(data_df=data_df)
    # I'm going to keep features I think would be most relevant
    # I'm a bit skepctical about fare as we have class, but fare would give more of an insight into how wealthy the passengers were
    data_df.index = data_df["PassengerId"]
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
