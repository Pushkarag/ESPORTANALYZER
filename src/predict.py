# src/predict.py
from pathlib import Path
import json
import joblib
import pandas as pd
from .feature_engineering import add_features, FEATURES_FOR_MODEL

MODEL_PATH = Path("models/best_model.pkl")
META_PATH  = Path("models/model_meta.json")

class ModelService:
    def __init__(self):
        self.model = joblib.load(MODEL_PATH)
        self.meta = json.loads(META_PATH.read_text())
        self.features = self.meta.get("features", FEATURES_FOR_MODEL)

    def _prepare(self, row_dict: dict) -> pd.DataFrame:
        base = pd.DataFrame([row_dict])
        aug = add_features(base)
        return aug[self.features].copy()

    def predict(self, row_dict: dict) -> float:
        X = self._prepare(row_dict)
        return float(self.model.predict(X)[0])
