# Beijing Air Quality Analysis and AQI Prediction

This folder contains Beijing air-quality data preparation, analysis, and model experiments. The active three-class XGBoost app is maintained in this directory; the former weather-enabled app in this folder has been retired.

## Project contents

| Path | Description |
| --- | --- |
| [`data/`](data/) | Beijing air-quality source data and retained analysis outputs |
| [`notebooks/DB_Connection.ipynb`](notebooks/DB_Connection.ipynb) | Loads station CSV data into MySQL tables |
| [`notebooks/etl_data_processor.ipynb`](notebooks/etl_data_processor.ipynb) | Combines and prepares hourly station data in `all_regions` |
| [`notebooks/hourly_to_daily_pollutant_conversion.ipynb`](notebooks/hourly_to_daily_pollutant_conversion.ipynb) | Builds the daily pollutant-and-weather data and updates `aqi_daily` |
| [`notebooks/clean_readings.ipynb`](notebooks/clean_readings.ipynb) | Validates and cleans the provided hourly CSV |
| [`app/aqi_labels.py`](app/aqi_labels.py) | Shared China AQI labeling and station-day split rules |
| [`notebooks/eda_beijing_air_quality.ipynb`](notebooks/eda_beijing_air_quality.ipynb) | Exploratory analysis and visualizations |
| [`notebooks/stats_analysis_beijing_air_quality.ipynb`](notebooks/stats_analysis_beijing_air_quality.ipynb) | Station-level descriptive and statistical analyses |
| [`notebooks/comprehensive_model_evaluation.ipynb`](notebooks/comprehensive_model_evaluation.ipynb) | Compares two-, three-, and six-class model results |
| [`notebooks/two_class_model_comparison.ipynb`](notebooks/two_class_model_comparison.ipynb), [`notebooks/three_class_model_comparison.ipynb`](notebooks/three_class_model_comparison.ipynb), [`notebooks/six_class_model_comparison.ipynb`](notebooks/six_class_model_comparison.ipynb) | Model-family comparisons for each AQI classification task |
| [`notebooks/xgboost_6_vs_3_class_comparison.ipynb`](notebooks/xgboost_6_vs_3_class_comparison.ipynb) | Compares XGBoost six-class and three-class next-day predictions |
| [`notebooks/xgboost_pollutants_only_vs_pollutants_weather_comparison.ipynb`](notebooks/xgboost_pollutants_only_vs_pollutants_weather_comparison.ipynb) | Tests whether weather improves three-class AQI prediction beyond pollutant-only inputs |
| [`weather_model_comparison.md`](weather_model_comparison.md) | Archived findings from the former weather-feature experiment |

## Current app

The active three-class app is [`app/app.py`](app/app.py), with its training notebook in [`notebooks/three_class_aqi.ipynb`](notebooks/three_class_aqi.ipynb). See [`README_3class.md`](README_3class.md) for setup and usage.

## Database preparation

For database-based analysis, run `DB_Connection.ipynb`, then `etl_data_processor.ipynb`. Run `hourly_to_daily_pollutant_conversion.ipynb` to prepare the 2013-2025 daily table used by the model notebooks.

## Further analysis

Run the relevant notebooks in `notebooks/` in Jupyter. Data exploration, statistics, and model comparisons are separate analyses and are not required just to launch the three-class app.
