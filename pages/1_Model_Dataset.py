from __future__ import annotations

import math
import os
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = Path(os.environ.get("AQI_DATASET_PATH", APP_DIR / "aqi_weather_model_dataset.csv"))
PAGE_SIZE = 50


@st.cache_data(show_spinner="Loading model dataset...")
def load_dataset(path: str, modified_at: float) -> pd.DataFrame:
    del modified_at
    return pd.read_csv(path)


def _go_to_previous_page() -> None:
    st.session_state["dataset_page"] -= 1


def _go_to_next_page() -> None:
    st.session_state["dataset_page"] += 1


st.set_page_config(page_title="Model dataset", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"],
    button[data-testid="collapsedControl"] {
        display: none;
    }
    div.st-key-top-navigation div[data-testid="stHorizontalBlock"] > div:nth-child(n+2) {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True, key="top-navigation"):
    brand, prediction_link, data_link = st.columns([4, 1, 1])
    with brand:
        st.markdown("**Beijing Air Quality · Model Dataset**")
    with prediction_link:
        st.page_link("pages/prediction.py", label="Prediction")
    with data_link:
        st.page_link("pages/1_Model_Dataset.py", label="Data")

st.title("Dataset used by the prediction model")
st.write(
    "Browse the daily station-level examples used to train and evaluate the "
    "next-day AQI models. Each row contains one day's pollutant and weather "
    "summaries and the target AQI information for the following day."
)
st.caption(
    "Features: six daily pollutant summaries plus same-day temperature, pressure, "
    "dew point, rain, wind speed, and circular wind-direction summaries. CO is "
    "in mg/m³; other pollutant values are in µg/m³."
)

if not DATASET_PATH.is_file():
    st.error(f"Model dataset file not found: {DATASET_PATH}")
    st.info(
        "Run train_weather_model_release.py from the repository root. It writes "
        "aqi_weather_model_dataset.csv beside the model artifact. To load a dataset saved elsewhere, set AQI_DATASET_PATH before "
        "starting Streamlit."
    )
    st.stop()

try:
    modified_at = DATASET_PATH.stat().st_mtime
    dataset = load_dataset(str(DATASET_PATH), modified_at)
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not read the model dataset at {DATASET_PATH}: {error}")
    st.stop()

if dataset.empty:
    st.warning("The model dataset file contains no rows.")
    st.stop()

total_rows = len(dataset)
total_pages = math.ceil(total_rows / PAGE_SIZE)
st.metric("Model dataset rows", f"{total_rows:,}")

st.session_state.setdefault("dataset_page", 1)
st.session_state["dataset_page"] = min(
    max(int(st.session_state["dataset_page"]), 1),
    total_pages,
)

previous, page_selector, next_page = st.columns(
    [1, 2, 1],
    vertical_alignment="center",
)
with previous:
    st.button(
        "← Previous",
        disabled=st.session_state["dataset_page"] <= 1,
        width="stretch",
        on_click=_go_to_previous_page,
    )
with page_selector:
    st.selectbox(
        "Page",
        options=range(1, total_pages + 1),
        format_func=lambda page: f"Page {page} of {total_pages}",
        key="dataset_page",
        label_visibility="collapsed",
    )
with next_page:
    st.button(
        "Next →",
        disabled=st.session_state["dataset_page"] >= total_pages,
        width="stretch",
        on_click=_go_to_next_page,
    )

page = st.session_state["dataset_page"]
start = (page - 1) * PAGE_SIZE
end = min(start + PAGE_SIZE, total_rows)
st.caption(f"Showing rows {start + 1:,}–{end:,} of {total_rows:,}")
st.dataframe(dataset.iloc[start:end], use_container_width=True, hide_index=True)
st.download_button(
    "Download full dataset as CSV",
    data=DATASET_PATH.read_bytes(),
    file_name=DATASET_PATH.name,
    mime="text/csv",
)
