# Three-class next-day AQI model

This folder contains an explanatory Jupyter notebook that trains a next-day Beijing AQI classifier from MySQL, plus a Streamlit app to use the saved model. It reuses the shared labeling and feature-building modules in `air-quality-classification/app/`; it does not read CSV data.

## First-time setup (Windows PowerShell)

Run these commands from the repository root (the `Data_Science` folder):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r "air-quality-classification/requirements.txt"
python -m pip install jupyter ipykernel
python -m ipykernel install --user --name data-science --display-name "Python (Data Science)"
jupyter notebook
```

If `.venv` already exists, skip `python -m venv .venv`. If PowerShell reports that script execution is disabled when activating, use this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

In Jupyter, open `air-quality-classification/notebooks/three_class_aqi.ipynb`. If prompted to select an interpreter, choose the `.venv` Python at `.venv\Scripts\python.exe`, or the registered **Python (Data Science)** kernel. Run cells from the top using **Shift+Enter**. The first setup cell prints the project root and the exact shared AQI labeler path; check that it points inside `air-quality-classification\app`.

## Prepare the database

Set `DB_URL` in the same PowerShell window before launching Jupyter:

```powershell
$env:DB_URL = "mysql+pymysql://root:@localhost/air_quality"
```

The notebook reads `all_regions` when available, otherwise it combines the 12 station tables. If the database tables have not been prepared yet, run `air-quality-classification/notebooks/DB_Connection.ipynb` and then `air-quality-classification/notebooks/etl.ipynb`. Those preparation notebooks rely on the original PRSA files; the three-class training notebook itself reads only MySQL. If MySQL is unreachable or empty, the model notebook stops with a setup message and does not fall back to CSV.

## Learn the notebook flow

Run the sections in order:

1. **Imports and project paths:** loads dependencies and finds the shared project modules.
2. **Database and labels:** reads hourly rows from MySQL, normalizes `interpolated_columns`, applies the repository HJ 633 rules, and builds tomorrow’s AQI target.
3. **Chronological split:** reserves the last 20% of unique dates for a final test.
4. **Model tuning:** tunes Random Forest, XGBoost, and logistic regression on training dates only; preprocessing stays inside each model pipeline.
5. **Baselines and evaluation:** compares the models and probability ensemble with persistence and majority-class baselines. It reports date-bootstrap accuracy intervals, macro-F1, class precision/recall, confusion matrices, station accuracy, and within-one-class accuracy.
6. **Saved outputs:** writes evaluation CSVs into `air-quality-classification/data/` and the model into `air-quality-classification/models/`.

Class 0 is AQI 0–100, class 1 is 101–200, and class 2 is above 200. Read `air-quality-classification/data/results.csv` after the final cell completes; do not treat expected reference values in the report as measured results.

## Run the app

After the notebook finishes successfully, stop Jupyter or open another PowerShell window, activate `.venv`, and run from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run "air-quality-classification/app/app.py"
```

The app defaults to `air-quality-classification/models/3class_model.joblib`. Set `CLASS3_MODEL_PATH` if the model is stored elsewhere. The database page uses `DB_URL` and shows a readable message if MySQL is unavailable.

## Common fixes

- **`No module named 'aqi_labels'`:** make sure Jupyter opened the `Data_Science` repository root and the first setup cell was run. The expected file is `air-quality-classification/app/aqi_labels.py`.
- **Module imports work in PowerShell but fail in the notebook:** switch the notebook kernel to the `.venv` interpreter / **Python (Data Science)** kernel, then restart the kernel and run from the first cell.
- **`No module named sqlalchemy` or `xgboost`:** activate `.venv` and rerun `python -m pip install -r "air-quality-classification/requirements.txt"` in that same terminal; then restart the notebook kernel.
- **MySQL connection error:** verify the MySQL service, `DB_URL`, and the `air_quality` tables. Run the two database preparation notebooks if needed.
- **Do not type Python statements at `PS ...>`:** that is the PowerShell prompt. Python code goes into a notebook code cell; run it with **Shift+Enter**.

## Folder contents

- `notebooks/three_class_aqi.ipynb` — annotated training and evaluation notebook.
- `app/app.py` — single-file Streamlit app with prediction, results, and database views.
- `requirements.txt` — model and app dependencies.
- `REPORT.md` — plain-language method and interpretation guide.
- The notebook creates the model artifact and result CSVs after successful training.
