from pathlib import Path

import numpy as np
import pandas as pd

from data_loading import load_and_transform_data
from model_data import gradient_descent, predict


def convert_df_to_dict(data_df: pd.DataFrame):
    data_split_dict = data_df.to_dict(orient="split")
    return {data_split_dict["index"][idx]: data_split_dict["data"][idx] for idx in range(len(data_split_dict["data"]))}


def main(train_path: Path, test_path: Path):
    cols_to_normalise = ["Age", "Fare"]
    features = ['Survived', 'Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age_norm', 'Fare_norm']
    train_df = load_and_transform_data(file_path=train_path, cols_to_normalise=cols_to_normalise, features=features)
    # Some people are missing an age so I'm going to drop these for now - if it has a big impact I could replace nan values with an average
    train_df = train_df.dropna(how="any")
    # Not including survived as its not in the test data - thats what we're predicting for
    test_df = load_and_transform_data(file_path=test_path, cols_to_normalise=cols_to_normalise, features=features[1:])
    y_train = np.array(train_df["Survived"].to_list())
    train_passenger_id_to_feature = convert_df_to_dict(data_df=train_df)
    x_train = np.array(list(train_passenger_id_to_feature.values()))
    initial_w = 0.01 * (np.random.rand(x_train.shape[1]) - 0.5)
    initial_b = 0
    w, b, J_history, _ = gradient_descent(X=x_train, y=y_train, w_in=initial_w, b_in=initial_b, alpha=0.001, num_iters=30000)
    db=1
    # Test the model with the training set
    test_train_prediction = predict(X=x_train, w=w, b=b)
    print('Train Accuracy: %f' % (np.mean(test_train_prediction == y_train) * 100))
    # Use the model to make predictions for the test set
    output_list = list()
    test_passenger_id_to_feature = convert_df_to_dict(data_df=test_df)
    x_test = np.array(list(test_passenger_id_to_feature.values()))
    y_test = predict(X=x_test, w=w, b=b)
    for idx, p_id in enumerate(test_passenger_id_to_feature["index"]):
        output_list.append({"PassengerId": p_id, "Survived": y_test[idx]})
    results_df = pd.DataFrame(output_list)
    results_df.to_csv("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/results.csv", index=False)




if __name__ == '__main__':
    test_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    main(train_path=train_file_path, test_path=test_file_path)
