import streamlit as st

prediction_page = st.Page(
    "pages/prediction.py",
    title="Next-day AQI prediction",
    default=True,
)
dataset_page = st.Page(
    "pages/1_Model_Dataset.py",
    title="Model dataset",
)

st.navigation([prediction_page, dataset_page], position="sidebar").run()
