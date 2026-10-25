import pandas as pd
import streamlit as st
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
st.title("Model results")
for fn,title in [("results.csv","Model and baseline comparison"),("per_class.csv","Per-class precision, recall and F1")]:
    st.subheader(title); p=HERE/fn
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True)
    else: st.info("Run all cells in three_class_aqi.ipynb to generate results.")
with st.expander("Pollutant and Weather Measurements Baselines", expanded=False):
    p=HERE/"aqi_category_breakpoints.csv"
    if p.exists():
        st.caption("Pollutant cutoffs show when that pollutant's AQI sub-index enters each class. Overall AQI is set by the highest pollutant sub-index. Weather measurements have no direct AQI clean/polluted cutoff.")
        st.dataframe(pd.read_csv(p).drop(columns=["Notes"], errors="ignore"),hide_index=True,use_container_width=True)
    else:
        st.info("Run all cells in three_class_aqi.ipynb to generate AQI category breakpoints.")
st.subheader("Confusion matrix counts")
p=HERE/"confusion_matrix.csv"
if p.exists():
    d=pd.read_csv(p); name=st.selectbox("Model",d.model.unique()); d=d[d.model==name]
    st.dataframe(d.pivot(index="actual_label",columns="predicted_label",values="count"),use_container_width=True)
else: st.info("Run all cells in three_class_aqi.ipynb to generate results.")
