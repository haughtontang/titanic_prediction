from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from data_loading import transform_data, get_median_for_entire_data_set, add_family_size, convert_sex_to_number, convert_location_to_number
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


def convert_df_to_dict(data_df: pd.DataFrame):
    data_split_dict = data_df.to_dict(orient="split")
    return {data_split_dict["index"][idx]: data_split_dict["data"][idx] for idx in range(len(data_split_dict["data"]))}


def load_and_transform_data(file_path, cols_to_normalise):
    data_df = pd.read_csv(file_path)
    return transform_data(data_df=data_df, cols_to_normalise=cols_to_normalise)


def load_data_skl(file_path: Path):
    data_df = pd.read_csv(file_path)
    data_df = convert_sex_to_number(data_df=data_df)
    data_df = convert_location_to_number(data_df=data_df)
    data_df.index = data_df["PassengerId"]
    return add_family_size(data_df=data_df)


def return_transformed_params(data_df: pd.DataFrame) -> np.ndarray:
    df_as_dict = convert_df_to_dict(data_df=data_df)
    return np.array(list(df_as_dict.values()))


def main(train_path: Path, test_path: Path, cols_to_normalise: list, features: list):
    # Loading in raw data and transforming
    scaler = StandardScaler()
    train_df = load_data_skl(file_path=train_path)
    test_df = load_data_skl(file_path=test_path)
    test_ids = test_df.index.to_list()
    for col in features:
        if col != "Survived":
            train_df, test_df = get_median_for_entire_data_set(test_df=test_df, train_df=train_df, column_name=col)

    train_df = train_df[features]
    features.remove("Survived")
    test_df = test_df[features]
    # Some people are missing an age so I'm going to drop these for now - if it has a big impact I could replace nan values with an average
    train_df = train_df.dropna(how="any")
    y_train = np.array(train_df["Survived"].to_list())
    train_df.drop(columns=["Survived"], inplace=True)

    x_train = return_transformed_params(data_df=train_df)
    x_test = return_transformed_params(data_df=test_df)

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # Define the model architecture
    model = Sequential([
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(x_train, y_train, epochs=1000)

    y_pred = model.predict(x_train)
    y_pred = (y_pred > 0.5).astype("int32")
    accuracy = accuracy_score(y_train, y_pred)
    print(accuracy * 100)
    test_prediction = model.predict(x_test)
    test_prediction = (test_prediction > 0.5).astype("int32")
    print(test_prediction)
    output_list = list()
    for idx, p_id in enumerate(test_ids):
        output_list.append({"PassengerId": p_id, "Survived": test_prediction[idx]})
    results_df = pd.DataFrame(output_list)
    results_df.to_csv("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/results_tf_nn.csv", index=False)


if __name__ == '__main__':
    test_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    columns_to_normalise = ["Age", "Fare"]
    # Original list I used
    # features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age_norm', 'Fare_norm']
    model_features = ['Survived', 'Pclass', 'Sex', 'Age', "family_size", "Fare", "Embarked"]
    main(train_path=train_file_path, test_path=test_file_path, cols_to_normalise=columns_to_normalise, features=model_features)
