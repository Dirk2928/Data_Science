import pandas as pd
import streamlit as st
from pathlib import Path
HERE=Path(__file__).resolve().parents[1]
st.title("Model results")
for fn,title in [("results.csv","Model and baseline comparison"),("per_class.csv","Per-class precision, recall and F1")]:
    st.subheader(title); p=HERE/fn
    if p.exists(): st.dataframe(pd.read_csv(p),use_container_width=True)
    else: st.info("Run train.py to generate results.")
st.subheader("Confusion matrix counts")
p=HERE/"confusion_matrix.csv"
if p.exists():
    d=pd.read_csv(p); name=st.selectbox("Model",d.model.unique()); d=d[d.model==name]
    st.dataframe(d.pivot(index="actual_label",columns="predicted_label",values="count"),use_container_width=True)
else: st.info("Run train.py to generate results.")
