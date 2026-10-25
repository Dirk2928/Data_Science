# 3-class next-day AQI model

This folder builds a three-class next-day Beijing AQI classifier directly from MySQL. It uses the repository's `Notebooks/aqi_labels.py` HJ 633 label rules and `weather_model_comparison.py` daily feature builder.

## Install and configure

From the project root:

```powershell
python -m pip install -r "3 class/requirements.txt"
$env:DB_URL = "mysql+pymysql://root:@localhost/air_quality"
```

Set `DB_URL` to your SQLAlchemy/PyMySQL connection URL; credentials are read only from the environment. The default is `mysql+pymysql://root:@localhost/air_quality`. Ensure MySQL contains `all_regions` or all 12 station tables. If unavailable, run `Notebooks/DB_Connection.ipynb` and `Notebooks/temp_table.ipynb` first. This workflow does not use CSVs.

## Train and run

```powershell
python "3 class/train.py"
python -m streamlit run "3 class/app.py"
```

Set `CLASS3_MODEL_PATH` to load a model artifact from another location. Training writes the model and evaluation tables beside this README.

## Folder layout

- `train.py` — database loading, shared label/feature construction, chronological model training and evaluation.
- `3class_model.joblib` — selected fitted model and metadata.
- `results.csv`, `per_class.csv`, `per_station.csv`, `confusion_matrix.csv` — held-out evaluation outputs.
- `app.py`, `pages/` — prediction, results, and live database sample pages.
- `REPORT.md` — explanation and conclusions.
- `requirements.txt` — Python dependencies.
