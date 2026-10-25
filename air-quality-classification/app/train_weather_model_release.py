"""Train a weather-enabled app artifact and its daily model dataset."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
from aqi_labels import AQI_CATEGORIES  # noqa: E402
from weather_model_comparison import POLLUTANTS, WEATHER, SOURCE, build_supervised_dataset  # noqa: E402
from weather_model_tuning import chronological_training_data  # noqa: E402

FEATURES = POLLUTANTS + WEATHER
MODEL_PATH = DATA_DIR / "aqi_next_day_weather_random_forest.joblib"
DATASET_PATH = DATA_DIR / "aqi_weather_model_dataset.csv"
TUNING_PATH = DATA_DIR / "weather_model_tuning_results.csv"


def best_params(feature_set: str) -> dict:
    results = pd.read_csv(TUNING_PATH)
    choices = results[results.feature_set.eq(feature_set)]
    best = choices.loc[choices.mean_cv_macro_f1.idxmax()]
    return {"n_estimators": int(best.n_estimators),
            "max_depth": None if pd.isna(best.max_depth) else int(best.max_depth),
            "min_samples_leaf": int(best.min_samples_leaf)}


def pipeline(kind: str, params: dict):
    estimator = RandomForestRegressor if kind == "regressor" else RandomForestClassifier
    settings = dict(params, random_state=42, n_jobs=-1)
    if kind != "regressor":
        settings["class_weight"] = "balanced_subsample"
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     (kind, estimator(**settings))])


def main() -> None:
    data = build_supervised_dataset(pd.read_csv(SOURCE))
    train, holdout, _ = chronological_training_data(data)
    category_params = best_params("pollutant_plus_weather")
    pollutant_params = best_params("pollutant_only")

    # Evaluate additional heads on the same untouched holdout before fitting release models.
    aqi_eval = pipeline("regressor", category_params).fit(train[FEATURES], train.target_AQI)
    main_eval = pipeline("classifier", category_params).fit(
        train[FEATURES], train.target_main_pollutant)
    extra_evaluation = {
        "aqi_mae": float(mean_absolute_error(holdout.target_AQI,
                                               aqi_eval.predict(holdout[FEATURES]))),
        "main_pollutant_accuracy": float(accuracy_score(
            holdout.target_main_pollutant, main_eval.predict(holdout[FEATURES]))),
    }

    # Refit all three heads on every eligible historical example for the app artifact.
    category_model = pipeline("classifier", category_params).fit(
        data[FEATURES], data.target_category)
    aqi_regressor = pipeline("regressor", category_params).fit(
        data[FEATURES], data.target_AQI.astype(float))
    main_pollutant_model = pipeline("classifier", category_params).fit(
        data[FEATURES], data.target_main_pollutant)

    category_labels = list(AQI_CATEGORIES.values())
    units = {
        "PM2.5": "µg/m³", "PM10": "µg/m³", "SO2": "µg/m³",
        "NO2": "µg/m³", "CO": "mg/m³", "O3": "µg/m³",
        "TEMP_mean": "°C", "TEMP_range": "°C", "PRES_mean": "hPa",
        "DEWP_mean": "°C", "RAIN_sum": "mm", "WSPM_mean": "m/s",
        "wd_sin": "unitless (-1 to 1)", "wd_cos": "unitless (-1 to 1)",
    }
    averaging = {
        **{p: "same-day summary; 24-hour mean except O3 daily maximum trailing 8-hour mean"
           for p in POLLUTANTS},
        "TEMP_mean": "same-day hourly mean", "TEMP_range": "same-day hourly maximum minus minimum",
        "PRES_mean": "same-day hourly mean", "DEWP_mean": "same-day hourly mean",
        "RAIN_sum": "same-day hourly total", "WSPM_mean": "same-day hourly mean",
        "wd_sin": "mean sine of same-day hourly wind directions",
        "wd_cos": "mean cosine of same-day hourly wind directions",
    }
    artifact = {
        "model": category_model,
        "aqi_regressor": aqi_regressor,
        "primary_pollutant_model": main_pollutant_model,
        "feature_requirements": {
            "features_in_order": FEATURES,
            "numeric_features": FEATURES,
            "required_features": POLLUTANTS,
            "optional_features": WEATHER,
            "units": units,
            "averaging_periods": averaging,
            "weather_input_note": (
                "Use completed same-day weather summaries. Wind inputs are the mean "
                "sine and cosine of hourly compass directions in degrees."
            ),
        },
        "aqi_standard": "HJ 633-2026 (effective 2026-03-01)",
        "standard_source": "https://www.mee.gov.cn/ywgz/fgbz/bz/bzwb/jcffbz/202602/t20260225_1144441.shtml",
        "category_mapping": {str(i): {"label": c}
                             for i, c in enumerate(category_labels, start=1)},
        "evaluation_metrics": {
            "category_accuracy": 0.39115450890292935,
            "category_precision_macro": 0.3725124887207722,
            "category_recall_macro": 0.43563691806101357,
            "category_f1_macro": 0.38979791339059267,
            **extra_evaluation,
        },
        "training_period": [str(data.date.min().date()), str(data.date.max().date())],
        "evaluation_training_period": [str(train.date.min().date()), str(train.date.max().date())],
        "test_period": [str(holdout.date.min().date()), str(holdout.date.max().date())],
        "best_parameters": {
            "category": category_params,
            "numeric_aqi": category_params,
            "main_pollutant": category_params,
            "pollutant_only_comparator": pollutant_params,
        },
        "sklearn_version": sklearn.__version__,
        "training_source": "bundled PRSA cleaned hourly dataset; same-day weather observations",
    }
    joblib.dump(artifact, MODEL_PATH, compress=3)
    data_columns = ["date", "station", *FEATURES,
                    "target_category", "target_AQI", "target_main_pollutant"]
    data[data_columns].to_csv(DATASET_PATH, index=False)
    print(json.dumps({"model_path": str(MODEL_PATH), "model_size_mb": round(MODEL_PATH.stat().st_size / 1e6, 1),
                      "dataset_path": str(DATASET_PATH), "rows": len(data),
                      "category_parameters": category_params,
                      "extra_holdout_metrics": extra_evaluation,
                      "feature_order": FEATURES}, indent=2))


if __name__ == "__main__":
    main()
