import pandas as pd

from pathlib import Path
from challenge.model import DelayModel

DATA_PATH = Path(__file__).parent / "data" / "data.csv"
MODEL_PATH = Path(__file__).parent / "challenge" / "model.pkl"

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    model = DelayModel()
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
