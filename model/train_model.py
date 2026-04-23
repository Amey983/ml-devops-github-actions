from pathlib import Path
import pickle

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "trained_model.pkl"


def train_and_save_model():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset = load_diabetes()
    model = LinearRegression()
    model.fit(dataset.data, dataset.target)

    with OUTPUT_PATH.open("wb") as model_file:
        pickle.dump(model, model_file)

    print("Model training complete. Saved trained_model.pkl")


if __name__ == "__main__":
    train_and_save_model()
