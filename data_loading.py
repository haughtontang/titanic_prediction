import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def convert_sex_to_number(data_df: pd.DataFrame):
    data_df["Embarked"] = data_df["Embarked"].replace("S", 1)
    data_df["Embarked"] = data_df["Embarked"].replace("C", 2)
    data_df["Embarked"] = data_df["Embarked"].replace("Q", 3)
    return data_df


def convert_location_to_number(data_df: pd.DataFrame):

    data_df["Sex"] = data_df["Sex"].replace("female", 0)
    data_df["Sex"] = data_df["Sex"].replace("male", 1)
    return data_df


def mean_normalization(data_df: pd.DataFrame, col_to_normalise: str):
    col_data = data_df[col_to_normalise]
    col_mean = col_data.mean
    divider = max(col_data) - min(col_data)
    data_df[f"{col_to_normalise}_norm"] = (col_data - col_mean) / divider
    return data_df


