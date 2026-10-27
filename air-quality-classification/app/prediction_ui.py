from pathlib import Path
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_DIR = APP_DIR.parent / "models"
MODEL = Path(os.environ.get("CLASS3_MODEL_PATH", MODEL_DIR / "3class_model.joblib"))
AQI_MODEL = Path(os.environ.get("AQI_REGRESSOR_PATH", MODEL_DIR / "aqi_regressor.joblib"))

FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP_mean", "TEMP_range",
    "PRES_mean", "DEWP_mean", "RAIN_sum", "WSPM_mean", "wd_sin", "wd_cos",
]
POLLUTANTS = FEATURES[:6]
REQUIRED_POLLUTANTS = [feature for feature in POLLUTANTS if feature != "O3"]
WEATHER = FEATURES[6:-2]
WEATHER_LABELS = {
    "TEMP_mean": "Average temperature",
    "TEMP_range": "Temperature difference: highest − lowest",
    "PRES_mean": "Average air pressure",
    "DEWP_mean": "Average dew point",
    "RAIN_sum": "Total rainfall",
    "WSPM_mean": "Average wind speed",
}
LABELS = {0: "Clean", 1: "Light-to-moderate polluted", 2: "Heavily polluted"}
RANGES = {0: "AQI 0–100", 1: "AQI 101–200", 2: "AQI 201+"}
UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³", "SO2": "µg/m³", "NO2": "µg/m³",
    "CO": "mg/m³", "O3": "µg/m³", "TEMP_mean": "°C", "TEMP_range": "°C",
    "PRES_mean": "hPa", "DEWP_mean": "°C", "RAIN_sum": "mm", "WSPM_mean": "m/s",
}
HELP = {
    "CO": "Today's 24-hour average carbon monoxide concentration, in mg/m³.",
    "O3": "Optional. Today's highest valid trailing 8-hour ozone average, in µg/m³. If blank, the model uses its training-data median.",
    "TEMP_mean": "Today's average air temperature.",
    "TEMP_range": "Today's highest temperature minus lowest temperature.",
    "PRES_mean": "Today's average atmospheric pressure.",
    "DEWP_mean": "Today's average dew point temperature.",
    "RAIN_sum": "Total rainfall today.",
    "WSPM_mean": "Today's average wind speed.",
}
WIND_DIRECTIONS = [
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
]
EXAMPLES = {
    "Clean": APP_DIR / "examples" / "clean.csv",
    "Moderate": APP_DIR / "examples" / "moderate.csv",
    "Heavily polluted": APP_DIR / "examples" / "heavily_polluted.csv",
}


def wind_features(direction):
    if direction == "Unknown / variable":
        return None, None
    angle = np.deg2rad(WIND_DIRECTIONS.index(direction) * 22.5)
    return float(np.sin(angle)), float(np.cos(angle))


def load_example(path):
    example = pd.read_csv(path).iloc[0]
    for feature in POLLUTANTS + WEATHER:
        st.session_state[feature] = float(example[feature])
    st.session_state["wind_direction"] = str(example["wind_direction"])


def show_prediction():
    st.title("Next-day 3-class prediction")
    st.write("Enter today's completed station-day summaries, or load an example. O3 and weather fields are optional.")

    if not MODEL.exists():
        st.error(f"Model not found: {MODEL}. Run all cells in three_class_aqi.ipynb first.")
        st.stop()
    try:
        artifact = joblib.load(MODEL)
        model = artifact["model"]
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        st.stop()
    if list(artifact.get("feature_order", FEATURES)) != FEATURES:
        st.error("Model feature order does not match prediction inputs.")
        st.stop()

    if not AQI_MODEL.exists():
        st.error(f"Numeric AQI model not found: {AQI_MODEL}. Run train_aqi_regressor.py first.")
        st.stop()
    try:
        aqi_artifact = joblib.load(AQI_MODEL)
        aqi_model = aqi_artifact["model"]
    except Exception as exc:
        st.error(f"Could not load numeric AQI model: {exc}")
        st.stop()
    if list(aqi_artifact.get("feature_order", [])) != FEATURES:
        st.error("Numeric AQI model feature order does not match prediction inputs.")
        st.stop()

    st.subheader("Try an example")
    example_columns = st.columns(3)
    for column, (name, path) in zip(example_columns, EXAMPLES.items()):
        with column:
            if st.button(f"Load {name.lower()}", key=f"load_{path.stem}", use_container_width=True):
                load_example(path)

    with st.expander("Download example CSV files"):
        download_columns = st.columns(3)
        for column, (name, path) in zip(download_columns, EXAMPLES.items()):
            with column:
                st.download_button(
                    f"Download {name.lower()}", data=path.read_bytes(),
                    file_name=path.name, mime="text/csv", key=f"download_{path.stem}",
                    use_container_width=True,
                )

    with st.form("prediction"):
        values = {}
        st.subheader("Air pollutants")
        st.caption("Daily concentrations. CO uses mg/m³; the others use µg/m³. O3 may be left blank.")
        pollutant_columns = st.columns(2)
        for index, feature in enumerate(POLLUTANTS):
            with pollutant_columns[index % 2]:
                values[feature] = st.number_input(
                    f"{feature} ({UNITS[feature]}){' — optional' if feature == 'O3' else ''}",
                    min_value=0.0, value=None,
                    step=0.1, format="%.2f", help=HELP.get(feature), key=feature,
                )

        st.subheader("Weather")
        st.caption("Use today's daily summaries. Leave any unavailable weather value blank.")
        weather_columns = st.columns(2)
        for index, feature in enumerate(WEATHER):
            with weather_columns[index % 2]:
                minimum = None if feature in {"TEMP_mean", "DEWP_mean"} else 0.0
                values[feature] = st.number_input(
                    f"{WEATHER_LABELS[feature]} ({UNITS[feature]})", min_value=minimum, value=None,
                    step=0.1, format="%.2f", help=HELP.get(feature), key=feature,
                )
        direction = st.selectbox(
            "Prevailing wind direction", ["Unknown / variable", *WIND_DIRECTIONS],
            help="Choose today's prevailing compass direction. Unknown values are estimated from training data.",
            key="wind_direction",
        )
        submitted = st.form_submit_button("Predict tomorrow", type="primary")

    if submitted:
        missing = [feature for feature in REQUIRED_POLLUTANTS if values[feature] is None]
        if missing:
            st.error(f"Enter a value for: {', '.join(missing)}.")
        else:
            values["wd_sin"], values["wd_cos"] = wind_features(direction)
            row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
            probabilities = model.predict_proba(row)[0]
            predicted_class = int(model.classes_[np.argmax(probabilities)])
            estimated_aqi = int(np.clip(np.rint(aqi_model.predict(row)[0]), 0, 500))

            probability_column, result_column = st.columns([1, 1], gap="large")
            with probability_column:
                st.subheader("Class probabilities")
                for class_id, label in LABELS.items():
                    probability = probabilities[list(model.classes_).index(class_id)]
                    st.metric(label, f"{probability:.1%}")
            with result_column:
                st.subheader("Tomorrow's prediction")
                st.metric("Estimated AQI", f"≈ {estimated_aqi}")
                st.write(f"**Class:** {LABELS[predicted_class]}")
                st.caption(
                    f"Class definition: {RANGES[predicted_class]}. The numeric estimate and "
                    "class come from separate models and can disagree."
                )
            mae = aqi_artifact.get("metrics", {}).get("mae")
            if mae is not None:
                st.caption(
                    f"Historical numeric holdout mean absolute error: {mae:.0f} AQI points. "
                    "The class model was about 65% accurate on its historical holdout. "
                    "Predictions are not health guidance."
                )
