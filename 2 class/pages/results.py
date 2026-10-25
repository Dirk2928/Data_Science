from pathlib import Path
import pandas as pd
import streamlit as st
HERE=Path(__file__).resolve().parents[1]
st.title("Model results")
for filename,title in [("results.csv","Holdout accuracy and baselines"),("per_class.csv","Per-class precision, recall and F1"),("per_station.csv","Accuracy by station")]:
    st.subheader(title); path=HERE/filename
    if path.exists(): st.dataframe(pd.read_csv(path),use_container_width=True)
    else: st.info("Run all cells in two_class_aqi.ipynb to generate evaluation results.")
st.subheader("Confusion matrix")
path=HERE/"confusion_matrix.csv"
if path.exists():
    matrix=pd.read_csv(path).pivot(index="actual",columns="predicted",values="count")
    st.dataframe(matrix,use_container_width=True)
else: st.info("Run all cells in two_class_aqi.ipynb to generate the confusion matrix.")
