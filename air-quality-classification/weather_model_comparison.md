# Weather Feature Ablation

## Question

Does adding same-day meteorological readings improve the next-day AQI category model?

## Data and feature construction

The comparison uses the bundled hourly cleaned data in `PRSA_Data_20130301-20170228/cleaned_dataset.csv` and the shared AQI labeling implementation in `app/aqi_labels.py`. The target and pollutant feature rules match `model_development_final.ipynb`: today's six daily pollutant summaries predict the same station's category on the following calendar day. The supervised data contains 17,376 station-days across 1,459 dates.

The added weather features are daily mean and range of temperature, mean pressure, mean dew point, total rain, mean wind speed, and sine/cosine encodings of the circular mean wind direction. Every daily aggregate requires at least 20 nonmissing hourly readings; missing feature values are median-imputed within each training fit. No future-day observed weather is joined to the inputs.

The hourly file has 418,753 rows. Aggregation produces 17,492 station-days; 17,383 have all eight weather features available, and the remaining feature gaps are imputed inside each training fit. The joined supervised dataset still has 17,376 rows because target validity and next-day alignment determine the final sample.

## Timing and data-quality checks

The source spans 2013-03-01 through 2017-02-28 across 12 stations. There are no duplicate station-hour keys. Weather and pollutant summaries are grouped on the same station and calendar date. The target join explicitly shifts the label date back exactly one calendar day, so a missing target date does not silently turn a later day into the next-day label.

All weather predictors use observations from the feature date only. With the selected completed-day prediction cutoff, those readings are available after that date's hourly observations finish; the pipeline does not use target-day weather. The source has very low hourly weather missingness: 1 missing value each for TEMP, PRES, and DEWP; none for RAIN or WSPM; and 8 for wind direction. Daily feature missingness ranges from 108/17,492 (0.62%) for numeric summaries to 109/17,492 (0.62%) for wind direction encodings. Those gaps are handled by training-only median imputation.

The cleaned source contains some upstream-interpolated weather readings. Provenance flags mark 0.09% of usable TEMP, PRES, and DEWP readings, 0.07% of WSPM, and 0.43% of wind direction as interpolated; rain has no flagged interpolated readings. The comparison retains these readings to match the existing target and pollutant preparation, which also uses the cleaned source. A sensitivity analysis excluding flagged hours would be useful before production use.

These are same-day *observations*. The experiment represents a prediction made after today's daily measurements are complete. For predictions issued earlier, use weather forecasts issued by that cutoff and evaluate forecast vintages as they would have been available in practice.

## Method

An initial exploratory comparison used the final notebook's selected Random Forest settings (`n_estimators=300`, `max_depth=16`, `min_samples_leaf=3`) for both feature sets, with the same imputation, class weighting, and random seed. Dates were split chronologically, with every station on a date kept in the same partition.

For the matched tuning step, each feature set was separately tuned on the original 80% training dates using the final notebook's eight-candidate grid and four expanding-window folds grouped by date. The latest 20% remained reserved for evaluation. Both searches use macro F1, median imputation, the same Random Forest class weighting, and the same seed.

| Feature set | Best parameters | Training CV macro F1 |
|---|---|---:|
| Pollutants only | 300 trees, max depth 16, minimum leaf 3 | 0.3111 |
| Pollutants + weather | 600 trees, max depth 16, minimum leaf 3 | 0.3294 |

The weather feature set improves the matched training CV macro F1 by 0.0183. All 16 candidate settings and their fold scores are in `weather_model_tuning_results.csv`.

## Tuned model holdout evaluation

After selecting the best configuration for each feature set from training-only CV, both models were fit on all 13,894 training station-days and evaluated on the same 3,482 holdout station-days (2016-05-11–2017-02-27).

| Metric | Pollutants only | Pollutants + weather | Change |
|---|---:|---:|---:|
| Accuracy | 36.82% | 39.12% | +2.30 pp |
| Macro precision | 0.3494 | 0.3725 | +0.0231 |
| Macro recall | 0.4207 | 0.4356 | +0.0150 |
| Macro F1 | 0.3668 | 0.3898 | +0.0230 |

Per-category recall changed as follows:

| AQI category | Pollutants only | Pollutants + weather | Change |
|---|---:|---:|---:|
| Excellent | 42.19% | 45.44% | +3.25 pp |
| Good | 36.37% | 33.71% | -2.66 pp |
| Lightly polluted | 31.76% | 39.29% | +7.53 pp |
| Moderately polluted | 39.20% | 39.20% | 0.00 pp |
| Heavily polluted | 30.24% | 36.23% | +5.99 pp |
| Severely polluted | 72.65% | 67.52% | -5.13 pp |

Weather improves the overall held-out metrics, especially recall for lightly and heavily polluted days, while recall falls for Good and Severely polluted days. The full precision/recall/F1 breakdown is in `weather_model_holdout_class_report.csv`; normalized confusion counts are in `weather_model_holdout_confusion.csv`.

## Results

| Evaluation window | Pollutant accuracy | With weather | Change | Pollutant macro F1 | With weather | Change |
|---|---:|---:|---:|---:|---:|---:|
| Original 80/20 holdout, exploratory fixed settings (2016-05-11–2017-02-27) | 36.82% | 40.06% | +3.25 pp | 0.3668 | 0.3997 | +0.0329 |
| Early (2015-07-24–2015-12-16) | 39.10% | 43.82% | +4.72 pp | 0.3495 | 0.4023 | +0.0528 |
| Middle (2015-12-17–2016-05-10) | 36.54% | 37.11% | +0.57 pp | 0.3484 | 0.3473 | -0.0011 |
| Late (2016-05-11–2016-10-04) | 40.70% | 45.79% | +5.08 pp | 0.3179 | 0.3674 | +0.0496 |

These are exploratory comparisons with one fixed parameter set; the early and middle windows are not clean out-of-time confirmations because that setting was selected using the later training period. The late window overlaps the original holdout. The matched tuned-model evaluation and stability check below are the primary evidence.

## Stability within the held-out period

The tuned models were fit once on the original training period and scored separately on two consecutive, non-overlapping blocks in the untouched holdout. Neither block influenced training or parameter selection.

| Holdout period | Pollutant accuracy | With weather | Pollutant macro F1 | With weather |
|---|---:|---:|---:|---:|
| 2016-05-11–2016-10-04 (1,732 rows) | 40.70% | 44.52% | 0.3179 | 0.3553 |
| 2016-10-05–2017-02-27 (1,750 rows) | 32.97% | 33.77% | 0.3043 | 0.3123 |

Weather improves both metrics in both blocks, with a much smaller gain in the later block. This is consistency within one held-out span, not independent replication across years or locations.

Across the 12 stations, weather improves macro F1 at 10 stations and is slightly lower at Dingling and Wanliu; accuracy improves or ties at 11 stations. Seasonal macro F1 improves in spring, summer, and autumn; winter is slightly lower (0.316 vs 0.319). Seasonal coverage is uneven: the holdout has 252 spring station-days, compared with 1,068–1,092 in the other seasons. These subgroup results are descriptive given the limited historical period. Detailed breakdowns are saved in `weather_model_stability_stations.csv` and `weather_model_stability_seasons.csv`.

## Interpretation

In this historical comparison, the tuned weather model improved the held-out category metrics and both metrics in each of the two held-out blocks. The gain is promising rather than established for current conditions: this is historical data from one city and winter results were slightly weaker.

These results describe the retired weather-model experiment. They are not measurements of the current three-class XGBoost app's performance.

## Archived workflow

This document preserves previously generated weather-comparison findings. The scripts and walkthrough that produced them have been removed; the results remain as historical analysis and cannot be regenerated from the current codebase.
