from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, inspect, text

from prediction_ui import show_prediction


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
STATIONS = [
    "aotizhongxin", "changping", "dingling", "dongsi", "guanyuan", "gucheng",
    "huairou", "nongzhanguan", "shunyi", "tiantan", "wanliu", "wanshouxigong",
]


def show_results() -> None:
    st.title("Model results")
    for filename, title in [
        ("results.csv", "Model and baseline comparison"),
        ("per_class.csv", "Per-class precision, recall and F1"),
    ]:
        st.subheader(title)
        path = DATA_DIR / filename
        if path.exists():
            st.dataframe(pd.read_csv(path), use_container_width=True)
        else:
            st.info("Run all cells in notebooks/three_class_aqi.ipynb to generate results.")

    with st.expander("Pollutant and Weather Measurements Baselines", expanded=False):
        path = DATA_DIR / "aqi_category_breakpoints.csv"
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
            st.info("Run all cells in notebooks/three_class_aqi.ipynb to generate AQI category breakpoints.")

    st.subheader("Confusion matrix counts")
    path = DATA_DIR / "confusion_matrix.csv"
    if path.exists():
        data = pd.read_csv(path)
        model_name = st.selectbox("Model", data.model.unique())
        selected = data[data.model == model_name]
        st.dataframe(
            selected.pivot(index="actual_label", columns="predicted_label", values="count"),
            use_container_width=True,
        )
    else:
        st.info("Run all cells in notebooks/three_class_aqi.ipynb to generate results.")


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


st.set_page_config(page_title="Next-day 3-class AQI", page_icon="🌤️", layout="wide")
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
