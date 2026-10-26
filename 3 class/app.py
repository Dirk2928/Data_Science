from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("CLASS3_MODEL_PATH", APP_DIR / "3class_model.joblib"))
FEATURES = [
    "PM2.5", "PM10", "SO2", "NO2", "CO", "O3", "TEMP_mean", "TEMP_range",
    "PRES_mean", "DEWP_mean", "RAIN_sum", "WSPM_mean", "wd_sin", "wd_cos",
]
LABELS = {0: "Clean", 1: "Light-to-moderate polluted", 2: "Heavily polluted"}
AQI_RANGES = {0: "AQI 0–100", 1: "AQI 101–200", 2: "AQI 201+"}
UNITS = {
    "PM2.5": "µg/m³", "PM10": "µg/m³", "SO2": "µg/m³", "NO2": "µg/m³",
    "CO": "mg/m³", "O3": "µg/m³", "TEMP_mean": "°C", "TEMP_range": "°C",
    "PRES_mean": "hPa", "DEWP_mean": "°C", "RAIN_sum": "mm", "WSPM_mean": "m/s",
    "wd_sin": "unitless (-1 to 1)", "wd_cos": "unitless (-1 to 1)",
}
STATIONS = [
    "aotizhongxin", "changping", "dingling", "dongsi", "guanyuan", "gucheng",
    "huairou", "nongzhanguan", "shunyi", "tiantan", "wanliu", "wanshouxigong",
]


def show_prediction() -> None:
    st.title("Next-day 3-class prediction")
    st.write(
        "Enter today's completed station-day summaries. Weather fields are optional "
        "and median-imputed when blank."
    )
    st.info(
        "Wind direction uses sine/cosine values. Enter a pair whose vector length is at most 1, "
        "such as (wd_sin=0, wd_cos=1) or (wd_sin=0.707, wd_cos=0.707). Do not enter 1 for both fields."
    )
    if not MODEL_PATH.exists():
        st.error(f"Model not found: {MODEL_PATH}. Run all cells in three_class_aqi.ipynb first.")
        return
    try:
        artifact = joblib.load(MODEL_PATH)
        model = artifact["model"]
    except Exception as error:
        st.error(f"Could not load model: {error}")
        return
    if list(artifact.get("feature_order", FEATURES)) != FEATURES:
        st.error("Model feature order does not match prediction inputs.")
        return

    with st.form("prediction"):
        values = {}
        left, right = st.columns(2)
        for index, feature in enumerate(FEATURES):
            with (left if index % 2 == 0 else right):
                minimum = -1.0 if feature in {"wd_sin", "wd_cos"} else (
                    None if feature in {"TEMP_mean", "DEWP_mean"} else 0.0
                )
                values[feature] = st.number_input(
                    f"{feature} ({UNITS[feature]})", min_value=minimum, value=None,
                    step=0.1, format="%.3f" if feature in {"wd_sin", "wd_cos"} else "%.2f",
                    key=feature,
                )
        submitted = st.form_submit_button("Predict tomorrow", type="primary")

    if not submitted:
        return
    errors = []
    for feature, value in values.items():
        if value is None and feature in FEATURES[:6]:
            errors.append(f"Enter a value for {feature}.")
        elif value is not None and feature in {"wd_sin", "wd_cos"} and not -1 <= value <= 1:
            errors.append(f"{feature} must be between -1 and 1.")
    if values["TEMP_range"] is not None and values["TEMP_range"] < 0:
        errors.append("TEMP_range cannot be negative.")
    if (values["wd_sin"] is not None and values["wd_cos"] is not None
            and np.hypot(values["wd_sin"], values["wd_cos"]) > 1.01):
        errors.append(
            "Invalid wind direction: wd_sin and wd_cos together must have length at most 1. "
            "For example, use (0, 1), (1, 0), or (0.707, 0.707)."
        )
    if errors:
        for error in errors:
            st.error(error)
        return

    row = pd.DataFrame([[values[feature] for feature in FEATURES]], columns=FEATURES)
    probabilities = model.predict_proba(row)[0]
    predicted = int(model.classes_[np.argmax(probabilities)])
    st.subheader(f"{LABELS[predicted]} — {AQI_RANGES[predicted]}")
    for class_id, label in LABELS.items():
        class_index = list(model.classes_).index(class_id)
        st.metric(f"{label} probability", f"{probabilities[class_index]:.1%}")
    st.caption("Caution: this model is about 65% accurate on its historical holdout and is not health guidance.")


def show_results() -> None:
    st.title("Model results")
    for filename, title in [
        ("results.csv", "Model and baseline comparison"),
        ("per_class.csv", "Per-class precision, recall and F1"),
    ]:
        st.subheader(title)
        path = APP_DIR / filename
        if path.exists():
            st.dataframe(pd.read_csv(path), use_container_width=True)
        else:
            st.info("Run all cells in three_class_aqi.ipynb to generate results.")

    with st.expander("Pollutant and Weather Measurements Baselines", expanded=False):
        path = APP_DIR / "aqi_category_breakpoints.csv"
        if path.exists():
            st.caption(
                "Pollutant cutoffs show when that pollutant's AQI sub-index enters each class. "
                "Overall AQI is set by the highest pollutant sub-index. Weather measurements "
                "have no direct AQI clean/polluted cutoff."
            )
            st.dataframe(
                pd.read_csv(path).drop(columns=["Notes"], errors="ignore"),
                hide_index=True, use_container_width=True,
            )
        else:
            st.info("Run all cells in three_class_aqi.ipynb to generate AQI category breakpoints.")

    st.subheader("Confusion matrix counts")
    path = APP_DIR / "confusion_matrix.csv"
    if path.exists():
        data = pd.read_csv(path)
        model_name = st.selectbox("Model", data.model.unique())
        selected = data[data.model == model_name]
        st.dataframe(
            selected.pivot(index="actual_label", columns="predicted_label", values="count"),
            use_container_width=True,
        )
    else:
        st.info("Run all cells in three_class_aqi.ipynb to generate results.")


def show_database() -> None:
    st.title("Database data")
    st.caption("Live sample from MySQL air_quality; no CSV files are used.")
    engine = None
    try:
        database_url = os.environ.get("DB_URL", "mysql+pymysql://root:@localhost/air_quality")
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as connection:
            tables = set(inspect(connection).get_table_names())
            if "aqi_daily" in tables:
                source = "aqi_daily"
                stations = pd.read_sql_query(
                    text("SELECT DISTINCT station FROM aqi_daily ORDER BY station"), connection
                ).station.dropna().tolist()
            elif "all_regions" in tables:
                source = "all_regions"
                stations = pd.read_sql_query(
                    text("SELECT DISTINCT station FROM all_regions ORDER BY station"), connection
                ).station.dropna().tolist()
            else:
                source = [station for station in STATIONS if station in tables]
                if not source:
                    raise RuntimeError("No aqi_daily, all_regions, or station tables found.")
                stations = source

        choice = st.selectbox("Station", ["All stations", *stations])
        with engine.connect() as connection:
            if isinstance(source, str):
                count_query = (
                    f"SELECT COUNT(*) FROM `{source}`" if choice == "All stations"
                    else f"SELECT COUNT(*) FROM `{source}` WHERE station=:station"
                )
                count_params = {} if choice == "All stations" else {"station": choice}
                total_rows = connection.execute(text(count_query), count_params).scalar_one()
            else:
                choices = source if choice == "All stations" else [choice]
                table_counts = {
                    table: connection.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one()
                    for table in choices
                }
                total_rows = sum(table_counts.values())

            total_pages = (total_rows + 49) // 50
            page = st.number_input(
                "Page", min_value=1, max_value=max(1, total_pages), value=1, step=1
            )
            st.caption(f"Page {int(page)} of {total_pages} · {total_rows:,} rows total (50 rows per page)")
            offset = (int(page) - 1) * 50
            if isinstance(source, str):
                query = (
                    f"SELECT * FROM `{source}` LIMIT :lim OFFSET :off" if choice == "All stations"
                    else f"SELECT * FROM `{source}` WHERE station=:station LIMIT :lim OFFSET :off"
                )
                params = {"lim": 50, "off": offset}
                if choice != "All stations":
                    params["station"] = choice
                frame = pd.read_sql_query(text(query), connection, params=params)
            else:
                parts, remaining_offset, remaining_rows = [], offset, 50
                for table in choices:
                    table_count = table_counts[table]
                    if remaining_offset >= table_count:
                        remaining_offset -= table_count
                        continue
                    one = pd.read_sql_query(
                        text(f"SELECT * FROM `{table}` LIMIT :lim OFFSET :off"), connection,
                        params={"lim": remaining_rows, "off": remaining_offset},
                    )
                    if "station" not in one:
                        one["station"] = table
                    parts.append(one)
                    remaining_rows -= len(one)
                    remaining_offset = 0
                    if remaining_rows == 0:
                        break
                frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

        st.write(f"Source: {source if isinstance(source, str) else 'station tables'}")
        st.dataframe(frame, use_container_width=True)
    except Exception as error:
        st.error(
            "Unable to read MySQL air_quality. Prepare the database tables and check DB_URL."
        )
        st.caption(str(error))
    finally:
        if engine is not None:
            engine.dispose()


st.set_page_config(page_title="Next-day 3-class AQI", page_icon="🌤️")
prediction_page = st.Page(
    show_prediction,
    title="Next-day 3-class prediction",
    default=True,
)
results_page = st.Page(show_results, title="Model results")
database_page = st.Page(show_database, title="Database data")
st.navigation(
    [prediction_page, results_page, database_page],
    position="sidebar",
).run()
