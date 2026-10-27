"""Train a numeric next-day Beijing AQI estimate from the bundled daily dataset.

The classifier is trained separately by three_class_aqi.ipynb. This script uses the
same 14 input columns and a chronological holdout to evaluate numeric error.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "data" / "weather_augmented_model_dataset.csv"
OUTPUT = HERE / "models" / "aqi_regressor.joblib"
FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP_mean", "TEMP_range",
    "PRES_mean", "DEWP_mean", "RAIN_sum", "WSPM_mean", "wd_sin", "wd_cos",
]


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(
            n_estimators=250, max_depth=3, learning_rate=0.05,
            objective="reg:squarederror", random_state=42, n_jobs=1,
        )),
    ])


def main():
    data = pd.read_csv(SOURCE, parse_dates=["date"])
    data["target_AQI"] = pd.to_numeric(data["target_AQI"], errors="coerce")
    data = data.dropna(subset=["date", "target_AQI"]).sort_values(["date", "station"])
    dates = np.sort(data["date"].unique())
    cutoff = dates[int(len(dates) * 0.8)]
    train = data[data["date"] < cutoff]
    holdout = data[data["date"] >= cutoff]

    evaluation_model = make_model().fit(train[FEATURES], train["target_AQI"])
    estimates = evaluation_model.predict(holdout[FEATURES])
    metrics = {
        "mae": float(mean_absolute_error(holdout["target_AQI"], estimates)),
        "rmse": float(np.sqrt(mean_squared_error(holdout["target_AQI"], estimates))),
        "holdout_start": str(holdout["date"].min().date()),
        "holdout_end": str(holdout["date"].max().date()),
        "holdout_rows": len(holdout),
    }
    final_model = make_model().fit(data[FEATURES], data["target_AQI"])
    joblib.dump({
        "model": final_model,
        "feature_order": FEATURES,
        "metrics": metrics,
        "training_rows": len(data),
        "source": str(SOURCE.relative_to(HERE)),
    }, OUTPUT)
    print(f"Saved {OUTPUT}")
    print(f"Chronological holdout: MAE {metrics['mae']:.1f} AQI, RMSE {metrics['rmse']:.1f} AQI")


if __name__ == "__main__":
    main()
