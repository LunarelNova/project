import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from .config import TEST_SIZE, RANDOM_SEED

def load_data():
    df = pd.read_csv("data/heart.csv")

    # Split features and labels
    X = df.drop("target", axis=1)
    y = df["target"]

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Train-test split
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )