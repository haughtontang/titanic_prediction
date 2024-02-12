from pathlib import Path

import pandas as pd

from data_loading import load_and_transform_data


def convert_df_to_dict(data_df: pd.DataFrame):
    data_split_dict = data_df.to_dict(orient="split")
    return {data_split_dict["index"][idx]: data_split_dict["data"][idx] for idx in range(len(data_split_dict["data"]))}


def main(train_path: Path, test_path: Path):
    cols_to_normalise = ["Age", "Fare"]
    features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age_norm', 'Fare_norm']
    train_df = load_and_transform_data(file_path=train_path, cols_to_normalise=cols_to_normalise, features=features)
    # Not including survived as its not in the test data - thats what we're predicting for
    test_df = load_and_transform_data(file_path=test_path, cols_to_normalise=cols_to_normalise, features=features[1:])
    train_passenger_id_to_feature = convert_df_to_dict(data_df=train_df)
    test_passenger_id_to_feature = convert_df_to_dict(data_df=test_df)
    db = 1


if __name__ == '__main__':
    test_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    main(train_path=train_file_path, test_path=test_file_path)
