"""Matched temporal hyperparameter tuning for the two AQI feature sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from weather_model_comparison import POLLUTANTS, WEATHER, build_supervised_dataset, SOURCE

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
GRID = {
    "classifier__n_estimators": [300, 600],
    "classifier__max_depth": [None, 16],
    "classifier__min_samples_leaf": [1, 3],
}


def make_search(cv) -> GridSearchCV:
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            random_state=42, n_jobs=-1, class_weight="balanced_subsample"
        )),
    ])
    return GridSearchCV(
        pipeline, GRID, scoring="f1_macro", cv=cv, n_jobs=-1,
        refit=True, return_train_score=False, verbose=1,
    )


def chronological_training_data(data: pd.DataFrame):
    dates = np.array(sorted(data.date.unique()))
    cut = int(len(dates) * 0.80)
    train_dates, test_dates = dates[:cut], dates[cut:]
    train = data[data.date.isin(train_dates)].copy()
    test = data[data.date.isin(test_dates)].copy()
    date_codes = pd.factorize(train.date, sort=True)[0]
    unique_codes = np.unique(date_codes)
    splits = []
    for left, right in TimeSeriesSplit(n_splits=4).split(unique_codes):
        splits.append((np.flatnonzero(np.isin(date_codes, left)),
                       np.flatnonzero(np.isin(date_codes, right))))
    return train, test, splits


def tune_both(data: pd.DataFrame, *, verbose: bool = True):
    train, test, folds = chronological_training_data(data)
    results = []
    fitted = {}
    for label, features in (("pollutant_only", POLLUTANTS),
                            ("pollutant_plus_weather", POLLUTANTS + WEATHER)):
        if verbose:
            print(f"Tuning {label}: {len(train):,} train rows, {len(test):,} held-out rows")
        search = make_search(folds).fit(train[features], train.target_category)
        fitted[label] = search.best_estimator_
        for i, params in enumerate(search.cv_results_["params"]):
            item = {"feature_set": label, "candidate": i + 1,
                    "mean_cv_macro_f1": float(search.cv_results_["mean_test_score"][i]),
                    "std_cv_macro_f1": float(search.cv_results_["std_test_score"][i]),
                    **{k.replace("classifier__", ""): v for k, v in params.items()}}
            results.append(item)
        if verbose:
            print(json.dumps({"feature_set": label,
                              "best_params": {k.replace("classifier__", ""): v
                                              for k, v in search.best_params_.items()},
                              "best_cv_macro_f1": float(search.best_score_)}))
    cv_results = pd.DataFrame(results)
    return cv_results, fitted, train, test


def main() -> None:
    data = build_supervised_dataset(pd.read_csv(SOURCE))
    results, _, train, test = tune_both(data)
    out = DATA_DIR / "weather_model_tuning_results.csv"
    results.to_csv(out, index=False)
    print(f"Saved all candidate CV scores to {out}")
    print(f"Chronological training rows: {len(train):,}; held-out rows reserved for step 7: {len(test):,}")


if __name__ == "__main__":
    main()
