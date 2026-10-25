# Two-class next-day AQI model: measured results

## Target

The model predicts one of two labels for the next calendar day at the same station:

- **Clean:** AQI ≤ 100.
- **Polluted:** AQI > 100.

It uses today's six pollutant summaries and nine ERA5 weather summaries from `air_quality.aqi_daily`. The table contains 51,920 station-days; after requiring a valid next-day AQI, the notebook produced 50,266 supervised examples.

## Holdout evaluation

The first 80% of unique dates were used for training and the final 20% were kept as a chronological holdout. The training period was 2013-03-01 through 2022-08-31; the test period was 2022-09-01 through 2025-01-03, with 10,000 examples.

| Measure | Result |
|---|---:|
| XGBoost accuracy | 79.96% |
| 95% date-block bootstrap interval | 77.89%–82.13% |
| Majority baseline accuracy | 71.04% |
| Persistence baseline accuracy | 76.17% |
| Macro F1 | 0.754 |
| Clean recall | 86.68% |
| Polluted recall | 63.47% |

The model is 3.79 percentage points above the persistence baseline in overall accuracy. It identifies 63.47% of polluted test days, so some polluted days are still predicted as clean. Review `per_class.csv` and `confusion_matrix.csv` before using the model for warnings.

## Model selection

Four expanding-window, date-grouped folds selected XGBoost with 120 trees, maximum depth 5, and learning rate 0.06. The final test dates were not used for parameter selection.

These results are historical Beijing estimates, not official AQI readings or health advice. The weather features are observations from the completed current day, not next-day weather forecasts.
