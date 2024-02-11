import numpy as np
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression


def load_data(train_path: Path, test_path: Path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def convert_sex_to_number(data_df: pd.DataFrame):
    data_df["Sex"] = data_df["Sex"].replace("female", 0)
    data_df["Sex"] = data_df["Sex"].replace("male", 1)
    return data_df



def main(train_path: Path, test_path: Path):
    train_df, test_df = load_data(train_path=train_path, test_path=test_path)
    db = 1


if __name__ == '__main__':
    test_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    main(train_path=train_path, test_path=test_path)

