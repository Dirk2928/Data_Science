"""Evaluate the step-6 tuned feature sets on the untouched chronological holdout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.pipeline import Pipeline

from weather_model_comparison import POLLUTANTS, WEATHER, SOURCE, build_supervised_dataset
from weather_model_tuning import chronological_training_data
from aqi_labels import AQI_CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
TUNING_RESULTS = DATA_DIR / "weather_model_tuning_results.csv"
LABELS = list(AQI_CATEGORIES.values())


def main() -> None:
    data = build_supervised_dataset(pd.read_csv(SOURCE))
    train, holdout, _ = chronological_training_data(data)
    candidates = pd.read_csv(TUNING_RESULTS)
    summary, class_rows, confusion_rows = [], [], []
    for feature_set, features in (("pollutant_only", POLLUTANTS),
                                  ("pollutant_plus_weather", POLLUTANTS + WEATHER)):
        choices = candidates[candidates.feature_set.eq(feature_set)]
        best = choices.loc[choices.mean_cv_macro_f1.idxmax()]
        depth = None if pd.isna(best.max_depth) else int(best.max_depth)
        params = {"n_estimators": int(best.n_estimators), "max_depth": depth,
                  "min_samples_leaf": int(best.min_samples_leaf)}
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                **params, random_state=42, n_jobs=-1,
                class_weight="balanced_subsample")),
        ])
        model.fit(train[features], train.target_category)
        prediction = model.predict(holdout[features])
        truth = holdout.target_category
        summary.append({
            "feature_set": feature_set,
            "best_cv_macro_f1": float(best.mean_cv_macro_f1),
            "n_estimators": params["n_estimators"], "max_depth": params["max_depth"],
            "min_samples_leaf": params["min_samples_leaf"],
            "holdout_accuracy": accuracy_score(truth, prediction),
            "holdout_precision_macro": precision_score(truth, prediction, average="macro", zero_division=0),
            "holdout_recall_macro": recall_score(truth, prediction, average="macro", zero_division=0),
            "holdout_macro_f1": f1_score(truth, prediction, average="macro", zero_division=0),
            "holdout_rows": len(holdout),
        })
        report = classification_report(truth, prediction, labels=LABELS,
                                       output_dict=True, zero_division=0)
        for label in LABELS:
            class_rows.append({"feature_set": feature_set, "category": label,
                               "precision": report[label]["precision"],
                               "recall": report[label]["recall"],
                               "f1": report[label]["f1-score"],
                               "support": int(report[label]["support"])})
        matrix = confusion_matrix(truth, prediction, labels=LABELS)
        for actual_i, actual in enumerate(LABELS):
            for pred_i, predicted in enumerate(LABELS):
                confusion_rows.append({"feature_set": feature_set, "actual": actual,
                                       "predicted": predicted,
                                       "count": int(matrix[actual_i, pred_i])})
        print(feature_set, params, summary[-1])

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DATA_DIR / "weather_model_holdout_metrics.csv", index=False)
    pd.DataFrame(class_rows).to_csv(DATA_DIR / "weather_model_holdout_class_report.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(DATA_DIR / "weather_model_holdout_confusion.csv", index=False)
    print("Saved holdout summary, category report, and confusion matrices.")


if __name__ == "__main__":
    main()
