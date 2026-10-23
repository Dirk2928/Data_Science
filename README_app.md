# Next-day AQI prediction app

The prediction page accepts today's six completed pollutant summaries. It calculates today's overall AQI and main pollutant from pollutant inputs only, then shows tomorrow's most likely AQI category, predicted numeric AQI, and predicted main pollutant when those models are present in the saved artifact. Class probabilities are used internally to select the category and are not displayed. `app.py` explicitly registers the prediction and dataset pages with Streamlit navigation.

Inputs must use the training aggregation windows: 24-hour means for PM2.5, PM10, SO2, NO2, and CO; daily maximum trailing 8-hour average for O3. Units are µg/m³ except CO, which is mg/m³. Enter daily summaries, not single hourly readings.

## Run locally

From the project root, install the app dependencies and launch Streamlit:

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

By default, the app loads `aqi_next_day_random_forest.joblib` from the project root. To use another artifact, set `AQI_MODEL_PATH` before starting Streamlit. Run the training and export cells in `model_development_final.ipynb` to create the artifact.

The app navigation includes a dataset page that displays 50 model examples per page. The final notebook's export section also writes `aqi_model_dataset.csv` beside the model artifact. If the dataset file is missing, rerun the notebook through that section. Set `AQI_DATASET_PATH` before launching Streamlit to use a dataset file at another location.
