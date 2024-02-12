from pathlib import Path
from data_loading import load_data, convert_sex_to_number, mean_normalization


def main(train_path: Path, test_path: Path):
    train_df, test_df = load_data(train_path=train_path, test_path=test_path)
    cols_to_normalise = ["Age", "Fare"]
    for col in cols_to_normalise:
        train_df = mean_normalization(data_df=train_df, col_to_normalise=col)
        test_df = mean_normalization(data_df=test_df, col_to_normalise=col)
    train_df = convert_sex_to_number(data_df=train_df)
    test_df = convert_sex_to_number(data_df=test_df)
    db = 1


if __name__ == '__main__':
    test_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/test.csv")
    train_file_path = Path("/Users/donhaughton/Documents/PycharmProjects/titanic_prediction/data/raw_data/train.csv")
    main(train_path=train_file_path, test_path=test_file_path)
