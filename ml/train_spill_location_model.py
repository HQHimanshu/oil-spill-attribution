from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "backend" / "database" / "synthetic_ais_with_ship_data.csv"
MODEL_PATH = PROJECT_ROOT / "ml" / "models" / "spill_location_model.joblib"
METADATA_PATH = PROJECT_ROOT / "ml" / "models" / "spill_location_model_metadata.json"

TARGET_COLUMNS = ["spill_latitude", "spill_longitude"]
NUMERIC_FEATURES = [
    "LAT",
    "LON",
    "SOG",
    "COG",
    "Heading",
    "Length_m",
    "Width_m",
    "GrossTonnage",
    "Deadweight_t",
    "YearBuilt",
]
CATEGORICAL_FEATURES = [
    "ShipType",
    "Flag",
    "Status",
    "Destination",
    "spill_category",
    "spill_subcategory",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    missing = [col for col in TARGET_COLUMNS + FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC_FEATURES + TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=TARGET_COLUMNS + FEATURE_COLUMNS).copy()
    return df


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=1,
                ),
            ),
        ]
    )
    return model


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    predictions = model.predict(X_test)
    mae_latitude = mean_absolute_error(y_test["spill_latitude"], predictions[:, 0])
    mae_longitude = mean_absolute_error(y_test["spill_longitude"], predictions[:, 1])

    r2_latitude = r2_score(y_test["spill_latitude"], predictions[:, 0])
    r2_longitude = r2_score(y_test["spill_longitude"], predictions[:, 1])

    payload = {
        "mae_latitude": round(float(mae_latitude), 6),
        "mae_longitude": round(float(mae_longitude), 6),
        "r2_latitude": round(float(r2_latitude), 6),
        "r2_longitude": round(float(r2_longitude), 6),
        "mean_absolute_error_degrees": round(float((mae_latitude + mae_longitude) / 2.0), 6),
    }
    return payload


def main() -> None:
    df = load_dataset()
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMNS]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = build_model()
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METADATA_PATH.write_text(
        json.dumps(
            {
                "target_columns": TARGET_COLUMNS,
                "feature_columns": FEATURE_COLUMNS,
                "data_path": str(DATA_PATH),
                "model_path": str(MODEL_PATH),
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"model_path": str(MODEL_PATH), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
