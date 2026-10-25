from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
HERE=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ.get("CLASS3_MODEL_PATH",HERE/"3class_model.joblib"))
FEATURES=["PM2.5","PM10","SO2","NO2","CO","O3","TEMP_mean","TEMP_range","PRES_mean","DEWP_mean","RAIN_sum","WSPM_mean","wd_sin","wd_cos"]
LABELS={0:"Clean",1:"Light-to-moderate",2:"Heavy"}
RANGES={0:"AQI 0–100",1:"AQI 101–200",2:"AQI 201+"}
UNITS={"PM2.5":"µg/m³","PM10":"µg/m³","SO2":"µg/m³","NO2":"µg/m³","CO":"mg/m³","O3":"µg/m³","TEMP_mean":"°C","TEMP_range":"°C","PRES_mean":"hPa","DEWP_mean":"°C","RAIN_sum":"mm","WSPM_mean":"m/s","wd_sin":"unitless (-1 to 1)","wd_cos":"unitless (-1 to 1)"}
st.set_page_config(page_title="Next-day 3-class AQI",page_icon="🌤️")
st.title("Next-day 3-class prediction")
st.write("Enter today's completed station-day summaries. Weather fields are optional and median-imputed when blank.")
if not MODEL.exists(): st.error(f"Model not found: {MODEL}. Run train.py first."); st.stop()
try: artifact=joblib.load(MODEL); model=artifact["model"]
except Exception as e: st.error(f"Could not load model: {e}"); st.stop()
if list(artifact.get("feature_order",FEATURES))!=FEATURES: st.error("Model feature order does not match prediction inputs."); st.stop()
with st.form("prediction"):
    vals={}; left,right=st.columns(2)
    for i,f in enumerate(FEATURES):
        with (left if i%2==0 else right):
            default=None
            minimum=-1.0 if f in {"wd_sin","wd_cos"} else (None if f in {"TEMP_mean","DEWP_mean"} else 0.0)
            vals[f]=st.number_input(f"{f} ({UNITS[f]})",min_value=minimum,value=default,step=0.1,format="%.3f" if f in {"wd_sin","wd_cos"} else "%.2f",key=f)
    go=st.form_submit_button("Predict tomorrow",type="primary")
if go:
    errors=[]
    for f,v in vals.items():
        if v is None and f in FEATURES[:6]: errors.append(f"Enter a value for {f}.")
        elif v is not None and f in {"wd_sin","wd_cos"} and not -1<=v<=1: errors.append(f"{f} must be between -1 and 1.")
    if vals["TEMP_range"] is not None and vals["TEMP_range"]<0: errors.append("TEMP_range cannot be negative.")
    if vals["wd_sin"] is not None and vals["wd_cos"] is not None and np.hypot(vals["wd_sin"],vals["wd_cos"])>1.01: errors.append("Wind-direction vector length cannot exceed 1.")
    if errors:
        for e in errors: st.error(e)
    else:
        row=pd.DataFrame([[vals[f] for f in FEATURES]],columns=FEATURES)
        probs=model.predict_proba(row)[0]; pred=int(model.classes_[np.argmax(probs)])
        st.subheader(f"{LABELS[pred]} — {RANGES[pred]}")
        for i,label in LABELS.items(): st.metric(f"{label} probability",f"{probs[list(model.classes_).index(i)]:.1%}")
        st.caption("Caution: this model is about 65% accurate on its historical holdout and is not health guidance.")
