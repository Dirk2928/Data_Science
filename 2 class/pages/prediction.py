from pathlib import Path
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

HERE=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ.get("CLASS2_MODEL_PATH",HERE/"2class_model.joblib"))
FEATURES=["PM2.5","PM10","SO2","NO2","CO","O3","TEMP_mean","TEMP_range","PRES_mean","DEWP_mean","RH_mean","RAIN_sum","WSPM_mean","wd_sin","wd_cos"]
UNITS={"PM2.5":"µg/m³","PM10":"µg/m³","SO2":"µg/m³","NO2":"µg/m³","CO":"mg/m³","O3":"µg/m³","TEMP_mean":"°C","TEMP_range":"°C","PRES_mean":"hPa","DEWP_mean":"°C","RH_mean":"%","RAIN_sum":"mm","WSPM_mean":"m/s","wd_sin":"unitless (-1 to 1)","wd_cos":"unitless (-1 to 1)"}
st.set_page_config(page_title="Next-day Clean vs Polluted",page_icon="🌤️")
st.title("Next-day Clean vs Polluted")
st.write("Enter today's completed pollutant and daily weather summaries. Weather inputs can be left blank; the trained pipeline imputes missing values.")
if not MODEL.exists(): st.error(f"Model not found: {MODEL}. Run all cells in two_class_aqi.ipynb first."); st.stop()
try:
    artifact=joblib.load(MODEL); model=artifact["model"]; encoder=artifact["label_encoder"]
except Exception as error: st.error(f"Could not load model artifact: {error}"); st.stop()
if artifact.get("features")!=FEATURES: st.error("The saved model feature order does not match the app."); st.stop()
with st.form("binary_prediction"):
    values={}; left,right=st.columns(2)
    for i,feature in enumerate(FEATURES):
        with (left if i%2==0 else right):
            minimum=-1.0 if feature in {"wd_sin","wd_cos"} else (None if feature in {"TEMP_mean","DEWP_mean"} else 0.0)
            values[feature]=st.number_input(f"{feature} ({UNITS[feature]})",min_value=minimum,value=None,step=0.1,format="%.3f" if feature in {"wd_sin","wd_cos"} else "%.2f",key=feature)
    submitted=st.form_submit_button("Predict tomorrow",type="primary")
if submitted:
    errors=[]
    for feature in FEATURES[:6]:
        if values[feature] is None: errors.append(f"Enter a value for {feature}.")
    for feature in ["wd_sin","wd_cos"]:
        value=values[feature]
        if value is not None and not -1<=value<=1: errors.append(f"{feature} must be between -1 and 1.")
    if values["TEMP_range"] is not None and values["TEMP_range"]<0: errors.append("TEMP_range cannot be negative.")
    if values["wd_sin"] is not None and values["wd_cos"] is not None and np.hypot(values["wd_sin"],values["wd_cos"])>1.01: errors.append("Wind-direction sine/cosine vector length must be at most 1.")
    if errors:
        for error in errors: st.error(error)
    else:
        row=pd.DataFrame([[values[f] for f in FEATURES]],columns=FEATURES)
        probabilities=model.predict_proba(row)[0]
        code=int(model.predict(row)[0])
        prediction=encoder.inverse_transform([code])[0]
        st.subheader(f"Tomorrow is predicted to be {prediction}")
        for i,label in enumerate(encoder.classes_): st.metric(f"{label} probability",f"{probabilities[i]:.1%}")
        st.caption("Clean means AQI ≤ 100; Polluted means AQI > 100. This model is a historical estimate, not an official AQI report or health advice.")
