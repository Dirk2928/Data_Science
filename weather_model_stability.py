"""Temporal-block, station, and season stability analysis for tuned models."""
from __future__ import annotations

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
from aqi_labels import AQI_CATEGORIES  # noqa: E402
from weather_model_comparison import POLLUTANTS, WEATHER, SOURCE, build_supervised_dataset  # noqa: E402
from weather_model_tuning import chronological_training_data  # noqa: E402

LABELS = list(AQI_CATEGORIES.values())
TUNING = ROOT / "weather_model_tuning_results.csv"


def score_groups(data, predictions, group_col, group_name):
    rows = []
    for group, indices in data.groupby(group_col, sort=True).groups.items():
        idx = np.asarray(list(indices))
        truth, pred = data.loc[idx, "target_category"], predictions[idx]
        rows.append({"group_type": group_name, "group": group, "rows": len(idx),
                     "accuracy": accuracy_score(truth, pred),
                     "macro_f1": f1_score(truth, pred, labels=LABELS,
                                           average="macro", zero_division=0)})
    return rows


def main():
    data = build_supervised_dataset(pd.read_csv(SOURCE))
    train, holdout, _ = chronological_training_data(data)
    holdout = holdout.copy().reset_index(drop=True)
    dates = sorted(holdout.date.unique())
    midpoint = len(dates) // 2
    split_date = pd.Timestamp(dates[midpoint])
    holdout["time_block"] = np.where(holdout.date < split_date,
                                     "holdout_early", "holdout_late")
    holdout["season"] = holdout.date.dt.month.map({12: "Winter", 1: "Winter", 2: "Winter",
                                                   3: "Spring", 4: "Spring", 5: "Spring",
                                                   6: "Summer", 7: "Summer", 8: "Summer",
                                                   9: "Autumn", 10: "Autumn", 11: "Autumn"})
    candidates = pd.read_csv(TUNING)
    temporal, station_rows, season_rows = [], [], []
    for name, features in (("pollutant_only", POLLUTANTS),
                           ("pollutant_plus_weather", POLLUTANTS + WEATHER)):
        choices = candidates[candidates.feature_set.eq(name)]
        best = choices.loc[choices.mean_cv_macro_f1.idxmax()]
        params = {"n_estimators": int(best.n_estimators),
                  "max_depth": None if pd.isna(best.max_depth) else int(best.max_depth),
                  "min_samples_leaf": int(best.min_samples_leaf)}
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                **params, random_state=42, n_jobs=-1,
                class_weight="balanced_subsample")),
        ]).fit(train[features], train.target_category)
        prediction = model.predict(holdout[features])
        for block, group in holdout.groupby("time_block", sort=True):
            idx = group.index.to_numpy()
            temporal.append({"feature_set": name, "period": block,
                             "start": group.date.min().date(), "end": group.date.max().date(),
                             "rows": len(group),
                             "accuracy": accuracy_score(group.target_category, prediction[idx]),
                             "macro_f1": f1_score(group.target_category, prediction[idx],
                                                   labels=LABELS, average="macro", zero_division=0)})
        station_rows.extend({"feature_set": name, **r} for r in
                             score_groups(holdout, prediction, "station", "station"))
        season_rows.extend({"feature_set": name, **r} for r in
                            score_groups(holdout, prediction, "season", "season"))
        print(name, "parameters", params)

    pd.DataFrame(temporal).to_csv(ROOT / "weather_model_stability_periods.csv", index=False)
    pd.DataFrame(station_rows).to_csv(ROOT / "weather_model_stability_stations.csv", index=False)
    pd.DataFrame(season_rows).to_csv(ROOT / "weather_model_stability_seasons.csv", index=False)
    print(f"Holdout split at {split_date.date()}; wrote period, station, and season summaries.")


if __name__ == "__main__":
    main()
