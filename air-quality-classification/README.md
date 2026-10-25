# Beijing Air Quality Analysis and AQI Prediction

This project explores hourly air-quality measurements from monitoring stations in
Beijing and includes a Streamlit app that predicts the next day's air-quality
category from completed daily pollutant and meteorological summaries.

The analysis covers pollutant and weather patterns, station-level differences,
and statistical summaries. The modeling workflow calculates daily AQI labels,
trains and evaluates a next-day classifier, and can export a model artifact for
the app.

## Project contents

| Path | Description |
| --- | --- |
| [`data/`](data/) | Beijing hourly air-quality CSVs, including station data and combined datasets |
| [`notebooks/DB_Connection.ipynb`](notebooks/DB_Connection.ipynb) | Connects to MySQL, previews a station dataset, and loads station CSVs into tables |
| [`notebooks/temp_table.ipynb`](notebooks/temp_table.ipynb) | Combines and prepares station data in the `all_regions` MySQL table |
| [`notebooks/clean_readings.ipynb`](notebooks/clean_readings.ipynb) | Validates and cleans the provided `cleaned_dataset.csv` into `cleaned_dataset_validated.csv` |
| [`app/aqi_labels.py`](app/aqi_labels.py) | Shared AQI labeling and dataset-splitting logic for modeling notebooks |
| [`notebooks/eda_beijing_air_quality.ipynb`](notebooks/eda_beijing_air_quality.ipynb) | Exploratory data analysis and visualizations |
| [`notebooks/stats_analysis_beijing_air_quality.ipynb`](notebooks/stats_analysis_beijing_air_quality.ipynb) | Descriptive statistics and station-level statistical analyses |
| [`notebooks/sets.ipynb`](notebooks/sets.ipynb) | Calculates daily China AQI labels and data splits |
| [`notebooks/model_development.ipynb`](notebooks/model_development.ipynb), [`notebooks/model_development_revised.ipynb`](notebooks/model_development_revised.ipynb) | Earlier/alternative model experiments |
| [`notebooks/model_development_final.ipynb`](notebooks/model_development_final.ipynb) | Pollutant-only baseline training, evaluation, and artifact export |
| [`weather_model_comparison.md`](weather_model_comparison.md) | Weather feature ablation, held-out evaluation, and stability findings |
| [`notebooks/weather_feature_walkthrough.ipynb`](notebooks/weather_feature_walkthrough.ipynb) | Interactive weather feature construction, model comparison, and evaluation |
| [`app/weather_model_tuning.py`](app/weather_model_tuning.py), [`app/weather_model_evaluation.py`](app/weather_model_evaluation.py), [`app/weather_model_stability.py`](app/weather_model_stability.py) | Reproducible tuning, holdout evaluation, and stability analysis |
| [`app/train_weather_model_release.py`](app/train_weather_model_release.py) | Trains and exports the weather-enabled app artifact |
| [`app/app.py`](app/app.py) | Streamlit multipage entry point and navigation |
| [`app/pages/prediction.py`](app/pages/prediction.py) | AQI prediction interface |
| [`app/pages/1_Model_Dataset.py`](app/pages/1_Model_Dataset.py) | Paginated view of the model's daily training/evaluation examples |
| [`requirements-app.txt`](requirements-app.txt) | Python dependencies for running the app |
| [`README_app.md`](README_app.md) | Additional app usage details |

## Run the app

From the repository root, install the app dependencies and launch Streamlit:

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app/app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

The app loads `data/aqi_next_day_weather_random_forest.joblib` by default. The
generated model artifact is not included in the repository. To
create it, run the weather model tuning and release steps below. You can point
the app to a compatible artifact with `AQI_MODEL_PATH`.

## Prepare data and train the model

The app's weather-enabled model uses the bundled cleaned hourly dataset and does
not require MySQL. From the repository root, run:

```bash
python app/weather_model_tuning.py
python app/train_weather_model_release.py
```

The tuning script writes `data/weather_model_tuning_results.csv`. The release script
uses those selected parameters to train the category, numeric AQI, and main
pollutant models, then writes the ignored model artifact and
`data/aqi_weather_model_dataset.csv` used by the dataset browser. The older
`notebooks/model_development_final.ipynb` remains the pollutant-only baseline workflow.

The dataset page displays 50 rows per page. To use a different CSV, set
`AQI_DATASET_PATH` before starting Streamlit.

## Prediction inputs

Enter completed daily summaries, not individual hourly sensor readings:

- PM2.5, PM10, SO2, NO2, and CO: 24-hour means.
- O3: daily maximum trailing 8-hour average.
- Weather: same-day mean temperature, daily temperature range, mean pressure,
  mean dew point, total rainfall, mean wind speed, and mean sine/cosine of hourly
  wind directions. These fields can be left blank; the model fills them with
  training-set medians.
- Units: pollutant concentrations are µg/m³ except CO (mg/m³); weather uses °C,
  hPa, mm, m/s, and unitless wind-direction components.

The app calculates today's AQI from the supplied pollutant values and predicts
the next calendar day's AQI category. Numeric AQI and main-pollutant predictions
are shown when those models are present in the saved artifact. See
[`README_app.md`](README_app.md) for more details.

## Data and model limitations

The supplied observations cover Beijing monitoring stations from 2013 to 2017.
Weather inputs are observed values for a completed day, not tomorrow's weather
forecast. Predictions trained on this historical dataset may not generalize to
current conditions, other locations, or different sensors. Model results are
experimental and should not be used as health or operational guidance.

## Further analysis

Run the relevant notebooks in `notebooks/` in Jupyter to explore the dataset.
The EDA, statistics, and alternate-model notebooks are independent analyses;
they are not required to launch the Streamlit app if a compatible model
artifact is already available.
