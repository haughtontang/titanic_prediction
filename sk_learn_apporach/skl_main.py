from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_loading import transform_data, get_median_for_entire_data_set
from sklearn.linear_model import LogisticRegression


def convert_df_to_dict(data_df: pd.DataFrame):
    data_split_dict = data_df.to_dict(orient="split")
    return {data_split_dict["index"][idx]: data_split_dict["data"][idx] for idx in range(len(data_split_dict["data"]))}


def load_and_transform_data(file_path, cols_to_normalise):
    data_df = pd.read_csv(file_path)
    return transform_data(data_df=data_df, cols_to_normalise=cols_to_normalise)


def return_transformed_params(data_df: pd.DataFrame):
    df_as_dict = convert_df_to_dict(data_df=data_df)
    return np.array(list(df_as_dict.values()))


def main(train_path: Path, test_path: Path, cols_to_normalise: list, features: list):
    # Loading in raw data and transforming
    train_df = load_and_transform_data(file_path=train_path, cols_to_normalise=cols_to_normalise)
    test_df = load_and_transform_data(file_path=test_path, cols_to_normalise=cols_to_normalise)
    train_df, test_df = get_median_for_entire_data_set(test_df=test_df, train_df=train_df)
    test_ids = test_df.index.to_list()

    train_df = train_df[features]
    features.remove("Survived")
    test_df = test_df[features]
    # Some people are missing an age so I'm going to drop these for now - if it has a big impact I could replace nan values with an average
    train_df = train_df.dropna(how="any")
    y_train = np.array(train_df["Survived"].to_list())
    train_df.drop(columns=["Survived"], inplace=True)

    x_train = return_transformed_params(data_df=train_df)
    x_test = return_transformed_params(data_df=test_df)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    accuracy = accuracy_score(y_train, y_pred)
    print(accuracy * 100)
    test_prediction = model.predict(x_test)
    output_list = list()
    for idx, p_id in enumerate(test_ids):
        output_list.append({"PassengerId": p_id, "Survived": test_prediction[idx]})
    results_df = pd.DataFrame(output_list)
    results_df.to_csv("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/results_skl.csv", index=False)


if __name__ == '__main__':
    test_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    columns_to_normalise = ["Age", "Fare"]
    # Original list I used
    # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age_norm', 'Fare_norm']
    model_features = ['Survived', 'Pclass', 'Sex', 'Age_norm', "family_size"]
    main(train_path=train_file_path, test_path=test_file_path, cols_to_normalise=columns_to_normalise, features=model_features)
