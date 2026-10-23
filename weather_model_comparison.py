"""Reproducible temporal ablation: pollutant-only vs pollutant + same-day weather.

Uses the bundled cleaned hourly dataset and the project's shared AQI label logic.
Weather inputs are daily observations, so this evaluates a completed-day use case;
they are not suitable for an earlier forecast unless replaced with weather forecasts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "Notebooks"))
from aqi_labels import build_daily_labels  # noqa: E402

SOURCE = ROOT / "PRSA_Data_20130301-20170228" / "cleaned_dataset.csv"
POLLUTANTS = ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
WEATHER = ["TEMP_mean", "TEMP_range", "PRES_mean", "DEWP_mean", "RAIN_sum", "WSPM_mean", "wd_sin", "wd_cos"]
PARAMS = dict(n_estimators=300, max_depth=16, min_samples_leaf=3,
              random_state=42, n_jobs=-1, class_weight="balanced_subsample")


def daily_weather(hourly: pd.DataFrame) -> pd.DataFrame:
    x = hourly.copy()
    x["date"] = pd.to_datetime(x[["year", "month", "day"]])
    for col in ["TEMP", "PRES", "DEWP", "RAIN", "WSPM"]:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    x["wd"] = x["wd"].astype("string").str.strip().str.upper()
    dirs = {d: i * 22.5 for i, d in enumerate(
        ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
         "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"])}
    angle = np.deg2rad(x["wd"].map(dirs))
    x["_wd_sin"], x["_wd_cos"] = np.sin(angle), np.cos(angle)
    g = x.groupby(["station", "date"], sort=False)
    out = g.agg(
        TEMP_mean=("TEMP", "mean"), TEMP_min=("TEMP", "min"), TEMP_max=("TEMP", "max"),
        PRES_mean=("PRES", "mean"), DEWP_mean=("DEWP", "mean"),
        RAIN_sum=("RAIN", "sum"), WSPM_mean=("WSPM", "mean"),
        _temp_n=("TEMP", "count"), _pres_n=("PRES", "count"),
        _dewp_n=("DEWP", "count"), _rain_n=("RAIN", "count"),
        _wspm_n=("WSPM", "count"),
    )
    out["TEMP_range"] = out.pop("TEMP_max") - out.pop("TEMP_min")
    # Circular mean handles north crossing the 0/360-degree boundary.
    out["wd_sin"] = g["_wd_sin"].mean()
    out["wd_cos"] = g["_wd_cos"].mean()
    # Require 20 observations for weather daily aggregates, consistent with pollutant means.
    out.loc[out["_temp_n"] < 20, ["TEMP_mean", "TEMP_range"]] = np.nan
    out.loc[out["_pres_n"] < 20, "PRES_mean"] = np.nan
    out.loc[out["_dewp_n"] < 20, "DEWP_mean"] = np.nan
    out.loc[out["_rain_n"] < 20, "RAIN_sum"] = np.nan
    out.loc[out["_wspm_n"] < 20, "WSPM_mean"] = np.nan
    wd_n = g["_wd_sin"].count()
    out.loc[wd_n < 20, ["wd_sin", "wd_cos"]] = np.nan
    out = out.drop(columns=["_rain_n", "_temp_n", "_pres_n", "_dewp_n", "_wspm_n"])
    return out.reset_index()


def make_pipeline() -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("classifier", RandomForestClassifier(**PARAMS))])


def build_supervised_dataset(hourly: pd.DataFrame) -> pd.DataFrame:
    """Join today's pollutant/weather features to the exact next calendar-day label."""
    daily, _ = build_daily_labels(hourly)
    daily = daily.reset_index()
    if "_date" in daily:
        daily = daily.rename(columns={"_date": "date"})
    if "_station_key" in daily:
        daily = daily.drop(columns="_station_key")
    daily["date"] = pd.to_datetime(daily["date"])
    pollutant_map = {"PM2.5": "PM2.5_24h_avg", "PM10": "PM10_24h_avg",
                     "SO2": "SO2_24h_avg", "NO2": "NO2_24h_avg",
                     "CO": "CO_mg_m3", "O3": "O3_8h_avg"}
    weather = daily_weather(hourly)
    iaqi_columns = [f"{pollutant}_iaqi" for pollutant in POLLUTANTS]
    valid_iaqi = daily[iaqi_columns].notna().any(axis=1)
    daily["main_pollutant"] = pd.Series(index=daily.index, dtype="object")
    daily.loc[valid_iaqi, "main_pollutant"] = (
        daily.loc[valid_iaqi, iaqi_columns].idxmax(axis=1).str.removesuffix("_iaqi")
    )
    daily_features = daily[["station", "date", *pollutant_map.values()]].rename(
        columns={v: k for k, v in pollutant_map.items()})
    daily_features = daily_features.merge(weather, on=["station", "date"], how="left", validate="one_to_one")
    targets = daily[["station", "date", "aqi_category", "aqi", "main_pollutant"]].copy()
    targets["date"] = targets["date"] - pd.Timedelta(days=1)
    targets = targets.rename(columns={"aqi_category": "target_category",
                                      "aqi": "target_AQI",
                                      "main_pollutant": "target_main_pollutant"})
    data = daily_features.merge(targets, on=["station", "date"], how="inner", validate="one_to_one")
    data = data.dropna(subset=["target_category"]).sort_values(["date", "station"]).reset_index(drop=True)
    return data


def main() -> None:
    hourly = pd.read_csv(SOURCE)
    data = build_supervised_dataset(hourly)
    output_columns = ["date", "station", *POLLUTANTS, *WEATHER,
                      "target_category", "target_AQI", "target_main_pollutant"]
    dataset_path = ROOT / "weather_augmented_model_dataset.csv"
    data[output_columns].to_csv(dataset_path, index=False)
    print(f"Saved joined supervised examples: {dataset_path} ({len(data):,} rows)")
    dates = np.array(sorted(data["date"].unique()))
    # Match the final notebook's 80/20 temporal holdout and add two expanding-window checks.
    splits = [("original_80_20", 0.80, 1.00), ("early_60_10", 0.60, 0.70),
              ("middle_70_10", 0.70, 0.80), ("late_80_10", 0.80, 0.90)]
    results = []
    for name, train_end, test_end in splits:
        i0, i1 = int(len(dates) * train_end), int(len(dates) * test_end)
        train_days, test_days = set(dates[:i0]), set(dates[i0:i1])
        train, test = data[data.date.isin(train_days)], data[data.date.isin(test_days)]
        row = {"split": name, "train_through": str(train.date.max().date()),
               "test_from": str(test.date.min().date()), "test_through": str(test.date.max().date()),
               "train_rows": len(train), "test_rows": len(test)}
        for tag, features in [("pollutant", POLLUTANTS), ("weather", POLLUTANTS + WEATHER)]:
            model = make_pipeline().fit(train[features], train.target_category)
            pred = model.predict(test[features])
            row[f"{tag}_accuracy"] = accuracy_score(test.target_category, pred)
            row[f"{tag}_macro_f1"] = f1_score(test.target_category, pred, average="macro", zero_division=0)
        row["accuracy_delta"] = row["weather_accuracy"] - row["pollutant_accuracy"]
        row["macro_f1_delta"] = row["weather_macro_f1"] - row["pollutant_macro_f1"]
        results.append(row)
        print(json.dumps(row))
    out = pd.DataFrame(results)
    out.to_csv(ROOT / "weather_model_comparison_results.csv", index=False)
    print("SUMMARY", json.dumps({
        "weather_accuracy_wins": int((out.accuracy_delta > 0).sum()),
        "weather_macro_f1_wins": int((out.macro_f1_delta > 0).sum()),
        "mean_accuracy_delta": float(out.accuracy_delta.mean()),
        "mean_macro_f1_delta": float(out.macro_f1_delta.mean()),
        "rows": len(data), "unique_dates": len(dates),
        "weather_features": WEATHER,
    }))


if __name__ == "__main__":
    main()
