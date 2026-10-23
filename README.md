# Beijing Air Quality Analysis and AQI Prediction

This project explores hourly air-quality measurements from monitoring stations in
Beijing and includes a Streamlit app that predicts the next day's air-quality
category from completed daily pollutant summaries.

The analysis covers pollutant and weather patterns, station-level differences,
and statistical summaries. The modeling workflow calculates daily AQI labels,
trains and evaluates a next-day classifier, and can export a model artifact for
the app.

## Project contents

| Path | Description |
| --- | --- |
| [`PRSA_Data_20130301-20170228/`](PRSA_Data_20130301-20170228/) | Beijing hourly air-quality CSVs, including station data and combined datasets |
| [`Notebooks/DB_Connection.ipynb`](Notebooks/DB_Connection.ipynb) | Connects to MySQL, previews a station dataset, and loads station CSVs into tables |
| [`Notebooks/temp_table.ipynb`](Notebooks/temp_table.ipynb) | Combines and prepares station data in the `all_regions` MySQL table |
| [`Notebooks/clean_readings.ipynb`](Notebooks/clean_readings.ipynb) | Validates and cleans the provided `cleaned_dataset.csv` into `cleaned_dataset_validated.csv` |
| [`Notebooks/aqi_labels.py`](Notebooks/aqi_labels.py) | Shared AQI labeling and dataset-splitting logic for modeling notebooks |
| [`Notebooks/eda_beijing_air_quality.ipynb`](Notebooks/eda_beijing_air_quality.ipynb) | Exploratory data analysis and visualizations |
| [`Notebooks/stats_analysis_beijing_air_quality.ipynb`](Notebooks/stats_analysis_beijing_air_quality.ipynb) | Descriptive statistics and station-level statistical analyses |
| [`Notebooks/sets.ipynb`](Notebooks/sets.ipynb) | Calculates daily China AQI labels and data splits |
| [`Notebooks/model_development.ipynb`](Notebooks/model_development.ipynb), [`Notebooks/model_development_revised.ipynb`](Notebooks/model_development_revised.ipynb) | Earlier/alternative model experiments |
| [`model_development_final.ipynb`](model_development_final.ipynb) | Final next-day model training, evaluation, and artifact export |
| [`app.py`](app.py) | Streamlit multipage entry point and navigation |
| [`pages/prediction.py`](pages/prediction.py) | AQI prediction interface |
| [`pages/1_Model_Dataset.py`](pages/1_Model_Dataset.py) | Paginated view of the model's daily training/evaluation examples |
| [`requirements-app.txt`](requirements-app.txt) | Python dependencies for running the app |
| [`README_app.md`](README_app.md) | Additional app usage details |

## Run the app

From the repository root, install the app dependencies and launch Streamlit:

```bash
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

The app loads `aqi_next_day_random_forest.joblib` from the repository root by
default. That generated model artifact is not included in the repository. To
create it, complete the database and model-training steps below. You can also
point the app to an artifact elsewhere by setting the `AQI_MODEL_PATH`
environment variable before launching it.

## Prepare data and train the model

The final training notebook reads the `all_regions` table from MySQL. If you
need to recreate that table:

1. Install and start MySQL, then configure the database connection used by the
   notebooks. See [`Installations`](Installations) for DBeaver setup notes. Do not
   commit database passwords or other credentials.
2. Run `Notebooks/DB_Connection.ipynb` to load the station CSV files into
   MySQL.
3. Run `Notebooks/temp_table.ipynb` to combine and prepare the station data
   into `all_regions`.
4. Run `model_development_final.ipynb` from top to bottom. It loads
   `all_regions`, builds daily labels, trains and evaluates the model, and
   exports the model artifact and `aqi_model_dataset.csv` used by the dataset
   browser.
5. Launch the app using the commands above.

The final notebook uses `AIR_QUALITY_DB_URL` when set; otherwise it uses its
configured local MySQL connection. Update that configuration to match your
database before running the notebook.

The dataset page displays 50 rows per page. To use a different CSV, set
`AQI_DATASET_PATH` before starting Streamlit.

## Prediction inputs

Enter completed daily summaries, not individual hourly sensor readings:

- PM2.5, PM10, SO2, NO2, and CO: 24-hour means.
- O3: daily maximum trailing 8-hour average.
- Units: µg/m³ except CO, which is mg/m³.

The app calculates today's AQI from the supplied pollutant values and predicts
the next calendar day's AQI category. Numeric AQI and main-pollutant predictions
are shown when those models are present in the saved artifact. See
[`README_app.md`](README_app.md) for more details.

## Data and model limitations

The supplied observations cover Beijing monitoring stations from 2013 to 2017.
Predictions trained on this historical dataset may not generalize to current
conditions, other locations, or different sensors. Model results are
experimental and should not be used as health or operational guidance.

## Further analysis

Run the relevant notebooks in `Notebooks/` in Jupyter to explore the dataset.
The EDA, statistics, and alternate-model notebooks are independent analyses;
they are not required to launch the Streamlit app if a compatible model
artifact is already available.
