# Two-class next-day AQI model

This folder contains a notebook and Streamlit app for predicting **Clean** vs. **Polluted** next-day air quality using XGBoost. The notebook reads daily pollutant and ERA5 summaries from MySQL table `air_quality.aqi_daily`; it does not read a CSV or modify the table.

## Classes and features

- **Clean:** next-day AQI ≤ 100.
- **Polluted:** next-day AQI > 100.
- Inputs: today's six daily pollutant summaries and nine ERA5 weather summaries. The app requires pollutant inputs; missing weather values are imputed by the trained pipeline.

## Setup

From the repository root, install dependencies in your Python environment:

```powershell
python -m pip install -r "2 class/requirements.txt"
python -m pip install jupyter ipykernel
```

Ensure MySQL database `air_quality` contains the `aqi_daily` table. Set the database URL before launching Jupyter if the default local credentials differ:

```powershell
$env:DB_URL = "mysql+pymysql://root:@localhost/air_quality"
```

Open `2 class/two_class_aqi.ipynb` and run all cells from top to bottom. The notebook uses a chronological 80/20 date split and four expanding-window training folds. It writes `2class_model.joblib`, metrics, per-class scores, a confusion matrix, and per-station results into this folder.

## Run the app

After the notebook has produced the model artifact:

```powershell
python -m streamlit run "2 class/app.py"
```

The app defaults to `2 class/2class_model.joblib`. Set `CLASS2_MODEL_PATH` to load the artifact from another location.

Accuracy should be interpreted with the class-level recall, date-bootstrap interval, and majority/persistence baselines; a binary label can hide errors near the AQI 100 threshold.
