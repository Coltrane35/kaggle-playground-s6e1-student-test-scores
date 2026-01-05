import pandas as pd
from src.config import DATA_DIR


def load_train_test():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def load_sample_submission():
    return pd.read_csv(DATA_DIR / "sample_submission.csv")


if __name__ == "__main__":
    train, test = load_train_test()
    sub = load_sample_submission()

    print("Train:", train.shape)
    print("Test :", test.shape)
    print("Submission template:", sub.shape)

    print("\nTrain columns:")
    print(train.columns.tolist())

    print("\nSubmission columns:")
    print(sub.columns.tolist())
