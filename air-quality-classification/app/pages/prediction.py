"""Simple Streamlit interface for the saved next-day AQI classifier"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import streamlit as st

APP_DIR=Path(__file__).resolve().parents[2]
MODEL_PATH=Path(os.environ.get("AQI_MODEL_PATH",APP_DIR/"data"/"aqi_next_day_weather_random_forest.joblib"))
POLLUTANT_FEATURES=["PM2.5","PM10","SO2","NO2","CO","O3"]
WEATHER_FEATURES=["TEMP_mean","TEMP_range","PRES_mean","DEWP_mean","RAIN_sum","WSPM_mean","wd_sin","wd_cos"]
FEATURES=POLLUTANT_FEATURES+WEATHER_FEATURES
DISPLAY_NAMES={"Excellent":"Excellent","Good":"Good","Lightly polluted":"Lightly polluted","Moderately polluted":"Moderately polluted","Heavily polluted":"Heavily polluted","Severely polluted":"Severely polluted","优":"Excellent","良":"Good","轻度污染":"Lightly polluted","中度污染":"Moderately polluted","重度污染":"Heavily polluted","严重污染":"Severely polluted"}
AQI_RANGE={"Excellent":"0–50","Good":"51–100","Lightly polluted":"101–150","Moderately polluted":"151–200","Heavily polluted":"201–300","Severely polluted":"301–500"}
AQI_BREAKPOINTS=np.array([0,50,100,150,200,300,400,500],dtype=float)
POLLUTANT_BREAKPOINTS={"PM2.5":[0,30,60,115,150,250,350,500],"PM10":[0,50,120,250,350,420,500,600],"SO2":[0,50,150,475,800,1600,2100,2620],"NO2":[0,40,80,180,280,565,750,940],"CO":[0,2,4,14,24,36,48,60],"O3":[0,100,160,215,265,800]}
O3_1H_BREAKPOINTS=[0,160,200,300,400,800,1000,1200]
BREAKPOINT_MAX={"PM2.5":500,"PM10":600,"SO2":2620,"NO2":940,"CO":60,"O3":800}
UNIT={feature:"µg/m³" for feature in POLLUTANT_FEATURES}; UNIT["CO"]="mg/m³"
UNIT.update({"TEMP_mean":"°C","TEMP_range":"°C","PRES_mean":"hPa","DEWP_mean":"°C","RAIN_sum":"mm","WSPM_mean":"m/s","wd_sin":"unitless (-1 to 1)","wd_cos":"unitless (-1 to 1)"})
AVERAGING={"PM2.5":"24-hour mean","PM10":"24-hour mean","SO2":"24-hour mean","NO2":"24-hour mean","CO":"24-hour mean","O3":"daily maximum trailing 8-hour mean","TEMP_mean":"same-day hourly mean","TEMP_range":"same-day hourly maximum minus minimum","PRES_mean":"same-day hourly mean","DEWP_mean":"same-day hourly mean","RAIN_sum":"same-day hourly total","WSPM_mean":"same-day hourly mean","wd_sin":"mean sine of same-day hourly wind directions","wd_cos":"mean cosine of same-day hourly wind directions"}

def _read_artifact(path:Path)->dict[str,Any]:
    artifact=joblib.load(path)
    if not isinstance(artifact,dict) or "model" not in artifact: raise ValueError("The model artifact must contain a 'model' entry.")
    model=artifact["model"]
    if not hasattr(model,"predict") or not hasattr(model,"predict_proba"): raise ValueError("The saved model must support predict() and predict_proba().")
    expected=artifact.get("feature_requirements",{}).get("features_in_order",FEATURES)
    if list(expected)!=FEATURES: raise ValueError(f"Artifact feature order {expected!r} does not match this app's pollutant and weather inputs.")
    names=getattr(model,"feature_names_in_",None)
    if names is not None and list(names)!=FEATURES: raise ValueError(f"Model was trained with features {list(names)!r}, not {FEATURES!r}.")
    return artifact

def _display_category(value:Any,mapping:dict[str,Any])->str:
    raw=str(value)
    if raw in DISPLAY_NAMES: return DISPLAY_NAMES[raw]
    info=mapping.get(raw,{}) if isinstance(mapping,dict) else {}
    if isinstance(info,dict):
        label=info.get("label") or info.get("english") or info.get("chinese")
        if label: return DISPLAY_NAMES.get(str(label),str(label))
    return raw

def validate_inputs(values:dict[str,float|None])->list[str]:
    errors=[]
    for feature,value in values.items():
        if value is None:
            if feature in POLLUTANT_FEATURES: errors.append(f"Enter a value for {feature}.")
            continue
        if not np.isfinite(float(value)):
            errors.append(f"{feature} must be a finite number.")
            continue
        number=float(value)
        if feature in POLLUTANT_FEATURES and number<0: errors.append(f"{feature} cannot be negative.")
        if feature in {"TEMP_range","PRES_mean","RAIN_sum","WSPM_mean"} and number<0: errors.append(f"{feature} cannot be negative.")
        if feature in {"wd_sin","wd_cos"} and not -1<=number<=1: errors.append(f"{feature} must be between -1 and 1.")
    if values.get("TEMP_range") is not None and values["TEMP_range"]<0: errors.append("Temperature range cannot be negative.")
    if values.get("wd_sin") is not None and values.get("wd_cos") is not None:
        if np.hypot(float(values["wd_sin"]),float(values["wd_cos"]))>1.01:
            errors.append("The wind-direction vector length cannot exceed 1. Enter the daily mean sine and cosine of hourly wind direction.")
    return errors

def _iaqi(value:float,breakpoints:list[float])->int:
    bp=np.asarray(breakpoints,dtype=float); aqi_bp=AQI_BREAKPOINTS[:len(bp)]
    return int(np.rint(np.interp(float(np.clip(value,bp[0],bp[-1])),bp,aqi_bp)))

def calculate_today_aqi(values:dict[str,float],o3_1h_max:float|None)->tuple[int,list[str],dict[str,int]]:
    indices={}
    for pollutant in POLLUTANT_FEATURES:
        if pollutant=="O3" and values[pollutant]>800:
            if o3_1h_max is None: raise ValueError("When O3's 8-hour mean exceeds 800 µg/m³, enter today's maximum 1-hour O3 value for the AQI fallback.")
            indices[pollutant]=_iaqi(o3_1h_max,O3_1H_BREAKPOINTS)
        else: indices[pollutant]=_iaqi(values[pollutant],POLLUTANT_BREAKPOINTS[pollutant])
    overall=max(indices.values())
    return overall,[p for p,v in indices.items() if v==overall],indices

st.set_page_config(page_title="Next-day AQI prediction",page_icon="🌤️",layout="centered")
st.sidebar.title("Beijing Air Quality")
st.sidebar.caption("AQI Predictor")
st.title("Next-day AQI category")
st.write("Enter today's completed pollutant and weather summaries to estimate the next calendar day's AQI. Use daily summaries, not instantaneous sensor readings.")
if not MODEL_PATH.exists():
    st.error(f"Saved model not found: {MODEL_PATH}"); st.info("Run train_weather_model_release.py from the repository root, or set AQI_MODEL_PATH to a compatible artifact."); st.stop()
try: artifact=_read_artifact(MODEL_PATH)
except Exception as error: st.error(f"Could not load the saved model: {error}"); st.stop()
model=artifact["model"]; feature_requirements=artifact.get("feature_requirements",{}); category_mapping=artifact.get("category_mapping",{})
st.caption(f"Model standard: {artifact.get('aqi_standard','not recorded')}")
with st.form("pollutant-inputs"):
    st.subheader("Today's measurements")
    st.caption("Pollutants: 24-hour means for PM2.5, PM10, SO2, NO2, and CO; daily maximum rolling 8-hour average for O3. Weather: completed same-day summaries. Weather inputs can be left blank when unavailable.")
    left,right=st.columns(2); input_values={}
    for i,feature in enumerate(POLLUTANT_FEATURES):
        with (left if i%2==0 else right):
            input_values[feature]=st.number_input(f"{feature} ({UNIT[feature]})",min_value=0.0,value=None,step=0.1,format="%.2f",key=f"input-{feature}")
    st.subheader("Today's meteorological summaries")
    st.caption("TEMP and DEWP in °C; pressure in hPa; rain in mm; wind speed in m/s. Wind direction uses the mean sine and cosine of hourly compass directions (degrees), each from -1 to 1.")
    weather_help={"TEMP_mean":"Mean hourly temperature for the day.","TEMP_range":"Daily maximum temperature minus minimum temperature.","PRES_mean":"Mean hourly atmospheric pressure.","DEWP_mean":"Mean hourly dew point.","RAIN_sum":"Sum of hourly rainfall readings for the day.","WSPM_mean":"Mean hourly wind speed.","wd_sin":"Arithmetic mean of sine(direction in radians) across valid hourly readings.","wd_cos":"Arithmetic mean of cosine(direction in radians) across valid hourly readings."}
    weather_left,weather_right=st.columns(2)
    for i,feature in enumerate(WEATHER_FEATURES):
        with (weather_left if i%2==0 else weather_right):
            minimum=-1.0 if feature in {"wd_sin","wd_cos"} else (None if feature in {"TEMP_mean","DEWP_mean"} else 0.0)
            input_values[feature]=st.number_input(f"{feature} ({UNIT[feature]})",min_value=minimum,value=None,step=0.1,format="%.3f" if feature in {"wd_sin","wd_cos"} else "%.2f",help=weather_help[feature],key=f"input-{feature}")
    o3_1h_max=st.number_input("Optional O₃ maximum 1-hour average (µg/m³)",min_value=0.0,value=None,step=1.0,format="%.1f",help="Only required to calculate today's AQI if the 8-hour O₃ value is above 800 µg/m³.",key="input-O3-1h-max")
    submitted=st.form_submit_button("Predict next-day category",type="primary",use_container_width=True)
if submitted:
    errors=validate_inputs(input_values)
    if input_values["O3"] is not None and input_values["O3"]>800 and o3_1h_max is None: errors.append("Enter the maximum 1-hour O₃ average because the 8-hour O₃ value exceeds 800 µg/m³.")
    if o3_1h_max is not None and (not np.isfinite(float(o3_1h_max)) or float(o3_1h_max)<0): errors.append("The optional 1-hour O₃ value must be finite and non-negative.")
    if errors:
        for message in errors: st.error(message)
    else:
        missing_weather=[feature for feature in WEATHER_FEATURES if input_values[feature] is None]
        if missing_weather: st.info(f"{len(missing_weather)} weather field(s) are blank; the model will use training-set median values for them.")
        for feature in POLLUTANT_FEATURES:
            if input_values[feature]>BREAKPOINT_MAX[feature]: st.warning(f"{feature} is above the highest concentration breakpoint ({BREAKPOINT_MAX[feature]} {UNIT[feature]}). The model can still predict, but this input may be outside its training range.")
        today_values={f:float(input_values[f]) for f in POLLUTANT_FEATURES}
        today_aqi,today_main,_=calculate_today_aqi(today_values,None if o3_1h_max is None else float(o3_1h_max))
        model_values={f:(today_values[f] if f in POLLUTANT_FEATURES else input_values[f]) for f in FEATURES}
        row=pd.DataFrame([[model_values[f] for f in FEATURES]],columns=FEATURES)
        probabilities=model.predict_proba(row)[0]; prediction=model.classes_[int(np.argmax(probabilities))]
        display_prediction=_display_category(prediction,category_mapping)
        aqi_model=artifact.get("aqi_regressor"); main_model=artifact.get("primary_pollutant_model")
        tomorrow_aqi=None if aqi_model is None else int(np.clip(np.ceil(aqi_model.predict(row)[0]),0,500))
        tomorrow_main=None if main_model is None else str(main_model.predict(row)[0])
        st.divider(); st.subheader("Prediction"); today,tomorrow=st.columns(2)
        with today:
            st.markdown("#### Today"); st.metric("Overall AQI",today_aqi)
            today_category=next((name for name,limits in AQI_RANGE.items() if today_aqi<=int(limits.split("–")[1])),"Severely polluted")
            st.write(f"**Category:** {today_category}"); st.write(f"**Main pollutant:** {', '.join(today_main)}")
        with tomorrow:
            st.markdown("#### Tomorrow"); st.metric("Predicted AQI category",display_prediction)
            if tomorrow_aqi is None:
                st.metric("Overall AQI",f"{AQI_RANGE.get(display_prediction,'—')} (category range)"); st.warning("This saved artifact has no numeric AQI regressor. Rerun the updated training notebook to add one.")
            else: st.metric("Predicted overall AQI",tomorrow_aqi)
            if tomorrow_main is None:
                st.write("**Main pollutant:** unavailable in this saved artifact"); st.warning("Rerun the updated training notebook to add next-day main-pollutant predictions.")
            else: st.write(f"**Predicted main pollutant:** {tomorrow_main}")
        st.caption("Tomorrow's category is the model's most likely class. Class probabilities are not displayed. These predictions are not official AQI reports.")
with st.expander("Input requirements"):
    st.write("Feature order expected by the saved model:",feature_requirements.get("features_in_order",FEATURES))
    averaging=feature_requirements.get("averaging_periods",AVERAGING)
    for feature in FEATURES: st.write(f"**{feature}** — {averaging.get(feature,AVERAGING[feature])}; {UNIT[feature]}.")
    st.write("Missing weather values are filled using training-set medians. Wind direction components are daily means of hourly sine and cosine values, with direction measured clockwise from north.")
