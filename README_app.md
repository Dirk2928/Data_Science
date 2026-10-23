# Next-day AQI prediction app

The prediction page accepts today's six completed pollutant summaries and eight same-day weather summaries. It calculates today's AQI using pollutant readings, then predicts tomorrow's AQI category, numeric AQI, and main pollutant. The weather fields are optional; blank values are median-imputed from training data.

Pollutant inputs use 24-hour means for PM2.5, PM10, SO2, NO2, and CO, plus the daily maximum trailing 8-hour average for O3. Concentration units are µg/m³ except CO (mg/m³). Weather summaries are mean/range temperature, mean pressure, mean dew point, total rain, mean wind speed, and the mean sine and cosine of hourly wind directions. Weather units are °C, hPa, mm, m/s, and unitless direction components.

Weather inputs describe an already completed day. The app does not fetch forecasts or use tomorrow's observed weather. For the wind-direction inputs, convert each hourly compass direction to degrees clockwise from north, calculate sine and cosine, then average each separately.

## Run locally

From the project root, install the app dependencies and launch Streamlit:

```powershell
python -m pip install -r requirements-app.txt
python -m streamlit run app.py
```

The app loads `aqi_next_day_weather_random_forest.joblib` by default. To build it, run `python weather_model_tuning.py` and then `python train_weather_model_release.py`. To use another compatible artifact, set `AQI_MODEL_PATH` before starting Streamlit.

The dataset page shows 50 rows per page from `aqi_weather_model_dataset.csv`, which the release script creates. Set `AQI_DATASET_PATH` before starting the app to use a different CSV.
