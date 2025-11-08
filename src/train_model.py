from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.feature_engineering import add_features, FEATURES_FOR_MODEL

PROCESSED = Path("data/processed/players_processed.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "best_model.pkl"
META_PATH  = MODEL_DIR / "model_meta.json"

def build_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED)
    df = add_features(df)

    # If you have a real label column, replace here:
    y = df["auction_value"].astype(float)

    X = df[FEATURES_FOR_MODEL].copy()
    return X, y

def train():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    X, y = build_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = list(X.columns)
    pre = ColumnTransformer(
        transformers=[("num", StandardScaler(with_mean=False), numeric_features)],
        remainder="drop",
        verbose_feature_names_out=False
    )

    candidates = [
        ("rf", RandomForestRegressor(random_state=42),
         {"rf__n_estimators":[200,400], "rf__max_depth":[None,10,20]}),
        ("xgb", XGBRegressor(random_state=42, n_estimators=500, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, tree_method="hist"),
         {"xgb__max_depth":[4,6,8]})
    ]

    best_model = None
    best_score = -np.inf
    best_name = None
    best_pipe = None

    for name, est, grid in candidates:
        pipe = Pipeline([("pre", pre), (name, est)])
        gs = GridSearchCV(pipe, grid, cv=3, scoring="r2", n_jobs=-1)
        gs.fit(X_train, y_train)
        print(f"{name} best R^2 (cv): {gs.best_score_:.4f}")
        if gs.best_score_ > best_score:
            best_score = gs.best_score_
            best_name = name
            best_pipe = gs.best_estimator_

    # Evaluate
    y_pred = best_pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    r2 = r2_score(y_test, y_pred)
    print(f"Test — MAE: {mae:.2f}  RMSE: {rmse:.2f}  R^2: {r2:.4f}  (model={best_name})")

    # Save
    joblib.dump(best_pipe, MODEL_PATH)
    meta = {
        "features": list(X.columns),
        "metrics": {"mae": float(mae), "rmse": float(rmse), "r2": float(r2)},
        "model": best_name
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved model -> {MODEL_PATH.resolve()}")
    print(f"Saved meta  -> {META_PATH.resolve()}")

if __name__ == "__main__":
    train()
