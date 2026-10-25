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

Open `three_class_aqi.ipynb` in Jupyter from the repository root and run its cells in order. The notebook reads MySQL, constructs labels and features using the shared project modules, tunes and evaluates the models, writes the results CSV files, and saves `3class_model.joblib`. It contains Markdown explanations alongside each workflow stage.

Then launch the Streamlit app from the repository root:

```powershell
python -m streamlit run "3 class/app.py"
```

Set `CLASS3_MODEL_PATH` to load a model artifact from another location. Training writes the model and evaluation tables beside this README.

## Folder layout

- `three_class_aqi.ipynb` — explained, end-to-end database loading, labeling, feature construction, training, evaluation, and artifact generation.
- `3class_model.joblib` — selected fitted model and metadata.
- `results.csv`, `per_class.csv`, `per_station.csv`, `confusion_matrix.csv` — held-out evaluation outputs.
- `app.py`, `pages/` — prediction, results, and live database sample pages.
- `REPORT.md` — explanation and conclusions.
- `requirements.txt` — Python dependencies.
