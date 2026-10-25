"""Train and evaluate a three-class next-day Beijing AQI model from MySQL."""
from __future__ import annotations
import json, os, sys, ast
sys.dont_write_bytecode=True
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/"Notebooks"))
sys.path.insert(0,str(ROOT))
from aqi_labels import build_daily_labels
from weather_model_comparison import build_supervised_dataset

STATIONS=["aotizhongxin","changping","dingling","dongsi","guanyuan","gucheng","huairou","nongzhanguan","shunyi","tiantan","wanliu","wanshouxigong"]
POLLUTANTS=["PM2.5","PM10","SO2","NO2","CO","O3"]
WEATHER=["TEMP_mean","TEMP_range","PRES_mean","DEWP_mean","RAIN_sum","WSPM_mean","wd_sin","wd_cos"]
FEATURES=POLLUTANTS+WEATHER
LABELS={0:"Clean",1:"Light-to-moderate",2:"Heavy"}


def _normalize_interpolated(values):
    """Give aqi_labels the comma-separated representation written by temp_table.ipynb."""
    def one(v):
        if v is None or (isinstance(v,float) and np.isnan(v)): return "None"
        if isinstance(v,(list,tuple,set)): return ", ".join(str(x) for x in v) if v else "None"
        s=str(v).strip()
        if s.startswith(("[","(")):
            try:
                parsed=ast.literal_eval(s)
                if isinstance(parsed,(list,tuple,set)): return ", ".join(map(str,parsed)) if parsed else "None"
            except (ValueError,SyntaxError): pass
        return "None" if s in ("", "[]", "None", "nan") else s
    return values.map(one)


def load_hourly():
    url=os.environ.get("DB_URL","mysql+pymysql://root:@localhost/air_quality")
    try:
        engine=create_engine(url)
        with engine.connect() as con:
            tables=set(inspect(con).get_table_names())
            if "all_regions" in tables:
                hourly=pd.concat(pd.read_sql_query(text("SELECT * FROM all_regions"),con,chunksize=50000),ignore_index=True)
                used=["all_regions"]
            else:
                available=[t for t in STATIONS if t in tables]
                if not available: raise RuntimeError("No all_regions or station tables found in air_quality.")
                parts=[]
                for table in available:
                    frame=pd.read_sql_query(text(f"SELECT * FROM `{table}`"),con)
                    if "station" not in frame: frame["station"]=table
                    parts.append(frame)
                hourly=pd.concat(parts,ignore_index=True); used=available
        engine.dispose()
    except Exception as exc:
        raise RuntimeError("MySQL air_quality is unreachable or has no usable source table. Run Notebooks/DB_Connection.ipynb and Notebooks/temp_table.ipynb first. No CSV fallback is used.") from exc
    if hourly.empty:
        raise RuntimeError("MySQL air_quality returned no rows. Run Notebooks/DB_Connection.ipynb and Notebooks/temp_table.ipynb first.")
    if "interpolated_columns" in hourly: hourly["interpolated_columns"]=_normalize_interpolated(hourly["interpolated_columns"])
    else: raise RuntimeError("Source table lacks interpolated_columns; rebuild all_regions with temp_table.ipynb.")
    print(f"Loaded {len(hourly):,} hourly rows from MySQL table(s): {', '.join(used)}")
    return hourly, used


def three_class_dataset(hourly):
    # Confirm source label counts against the repository's canonical HJ 633 labeler.
    daily,_=build_daily_labels(hourly)
    daily=daily.reset_index()
    if "_date" in daily: daily=daily.rename(columns={"_date":"date"})
    counts=daily["aqi_category"].value_counts(dropna=True).sort_index().to_dict()
    data=build_supervised_dataset(hourly)
    data["target_AQI"]=pd.to_numeric(data["target_AQI"],errors="coerce")
    # AQI class is defined directly from the canonical numeric AQI, with nulls removed.
    data=data.dropna(subset=["target_AQI"]).copy()
    data["target_class"]=np.select([data.target_AQI<=100,data.target_AQI<=200],[0,1],default=2).astype(int)
    data["date"]=pd.to_datetime(data["date"]).dt.normalize()
    data=data.sort_values(["date","station"]).reset_index(drop=True)
    print("Canonical daily HJ 633 labelled row counts by category:",json.dumps({str(k):int(v) for k,v in counts.items()}))
    print(f"Supervised next-day rows with valid AQI: {len(data):,}; dates: {data.date.nunique():,}")
    return data,counts


def _log_features(frame):
    x=np.asarray(frame,dtype=float).copy()
    # log1p for nonnegative pollutant concentrations, rainfall and wind speed only.
    for i in [0,1,2,3,4,5,10,11]: x[:,i]=np.log1p(np.maximum(x[:,i],0))
    return x


def make_models():
    rf=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",RandomForestClassifier(random_state=42,n_jobs=-1,class_weight="balanced_subsample"))])
    xgb=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",XGBClassifier(objective="multi:softprob",num_class=3,eval_metric="mlogloss",random_state=42,n_jobs=1,verbosity=0))])
    log=Pipeline([("imputer",SimpleImputer(strategy="median")),("log",FunctionTransformer(_log_features,validate=False)),("scale",StandardScaler()),("model",LogisticRegression(max_iter=2000,class_weight="balanced",random_state=42))])
    grids={"Random Forest":{"model__n_estimators":[150],"model__max_depth":[12],"model__min_samples_leaf":[3]},"XGBoost":{"model__n_estimators":[120],"model__max_depth":[3],"model__learning_rate":[0.06]},"Logistic regression":{"model__C":[1.0]}}
    return {"Random Forest":(rf,grids["Random Forest"]),"XGBoost":(xgb,grids["XGBoost"]),"Logistic regression":(log,grids["Logistic regression"]) }


def main():
    hourly,tables=load_hourly(); data,label_counts=three_class_dataset(hourly)
    if len(data)<100 or data.date.nunique()<10: raise RuntimeError("Not enough valid supervised dates to train and evaluate safely.")
    dates=np.array(sorted(data.date.unique())); cut=int(len(dates)*.8)
    train_dates,test_dates=dates[:cut],dates[cut:]
    train=data[data.date.isin(train_dates)].copy(); test=data[data.date.isin(test_dates)].copy()
    # Fold indices are constructed by unique date so no date is shared across train/validation.
    train_date_codes=pd.factorize(train.date,sort=True)[0]; unique_codes=np.arange(len(train_dates))
    folds=[(np.flatnonzero(np.isin(train_date_codes,a)),np.flatnonzero(np.isin(train_date_codes,b))) for a,b in TimeSeriesSplit(n_splits=4).split(unique_codes)]
    fitted={}; predictions={}; probabilities={}; records=[]
    for name,(pipe,grid) in make_models().items():
        search=GridSearchCV(pipe,grid,scoring="f1_macro",cv=folds,n_jobs=1,refit=True)
        search.fit(train[FEATURES],train.target_class)
        fitted[name]=search.best_estimator_; predictions[name]=search.predict(test[FEATURES]); probabilities[name]=search.predict_proba(test[FEATURES])
        print(f"{name} best CV macro-F1={search.best_score_:.4f}; params={search.best_params_}")
    probabilities["Ensemble"]=np.mean([probabilities[n] for n in ["Random Forest","XGBoost","Logistic regression"]],axis=0)
    predictions["Ensemble"]=np.argmax(probabilities["Ensemble"],axis=1)
    # Yesterday's AQI class is a persistence forecast; build via same day's canonical AQI.
    daily,_=build_daily_labels(hourly)
    today=daily.reset_index()
    if "_date" in today: today=today.rename(columns={"_date":"date"})
    today["date"]=pd.to_datetime(today.date).dt.normalize()
    today["today_class"]=np.select([pd.to_numeric(today.aqi,errors="coerce")<=100,pd.to_numeric(today.aqi,errors="coerce")<=200],[0,1],default=2)
    today["date"]=today["date"]+pd.Timedelta(days=1)
    today=today[["station","date","today_class"]]
    joined=test[["station","date","target_class","target_AQI"]].merge(today,on=["station","date"],how="left")
    joined["persistence"]=joined.today_class
    majority=int(train.target_class.value_counts().idxmax())
    predictions["Persistence"]=joined.persistence.fillna(majority).astype(int).to_numpy()
    predictions["Majority class"]=np.full(len(test),majority,dtype=int)
    rows=[]; per_class=[]
    y=test.target_class.to_numpy(); dates_test=test.date.to_numpy()
    rng=np.random.default_rng(42); unique_test=np.unique(dates_test)
    for name,pred in predictions.items():
        acc=accuracy_score(y,pred); boot=[]
        bydate={d:np.flatnonzero(dates_test==d) for d in unique_test}
        for _ in range(1000):
            sampled=rng.choice(unique_test,size=len(unique_test),replace=True); ix=np.concatenate([bydate[d] for d in sampled]); boot.append(accuracy_score(y[ix],pred[ix]))
        lo,hi=np.quantile(boot,[.025,.975]); p,r,f,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
        rows.append({"model":name,"accuracy":acc,"accuracy_ci95_low":lo,"accuracy_ci95_high":hi,"macro_f1":f1_score(y,pred,average="macro",zero_division=0),"accuracy_within_one_class":float(np.mean(np.abs(y-pred)<=1))})
        for i,label in LABELS.items(): per_class.append({"model":name,"class":i,"class_label":label,"precision":p[i],"recall":r[i],"f1":f[i],"support":int((y==i).sum())})
    results=pd.DataFrame(rows); results.to_csv(OUT/"results.csv",index=False)
    pd.DataFrame(per_class).to_csv(OUT/"per_class.csv",index=False)
    cm_rows=[]
    for name,pred in predictions.items():
        cm=confusion_matrix(y,pred,labels=[0,1,2])
        for i in range(3):
            for j in range(3): cm_rows.append({"model":name,"actual_class":i,"actual_label":LABELS[i],"predicted_class":j,"predicted_label":LABELS[j],"count":int(cm[i,j])})
    pd.DataFrame(cm_rows).to_csv(OUT/"confusion_matrix.csv",index=False)
    station_rows=[]
    for station,group in test.assign(_pred=predictions["Ensemble"]).groupby("station"):
        station_rows.append({"station":station,"rows":len(group),"accuracy":accuracy_score(group.target_class,group._pred)})
    pd.DataFrame(station_rows).to_csv(OUT/"per_station.csv",index=False)
    # Select on held-out accuracy as specified, then fit the chosen estimator on all labelled examples.
    candidates={**fitted,"Ensemble":None}
    model_name=max(candidates,key=lambda n:accuracy_score(y,predictions[n]))
    if model_name=="Ensemble":
        final=VotingClassifier(estimators=[("rf",fitted["Random Forest"]),("xgb",fitted["XGBoost"]),("lr",fitted["Logistic regression"])],voting="soft",weights=[1,1,1])
    else: final=clone(fitted[model_name])
    final.fit(data[FEATURES],data.target_class)
    best=results.loc[results.model==model_name].iloc[0].to_dict()
    artifact={"model":final,"model_name":model_name,"feature_order":FEATURES,"class_labels":LABELS,"training_period":[str(data.date.min().date()),str(data.date.max().date())],"test_period":[str(test.date.min().date()),str(test.date.max().date())],"metrics":best,"source_tables":tables,"canonical_label_counts":{str(k):int(v) for k,v in label_counts.items()},"training_rows":len(data)}
    joblib.dump(artifact,OUT/"3class_model.joblib")
    print("RESULTS",results.to_string(index=False)); print(f"Saved {model_name} model to {OUT/'3class_model.joblib'}")
    print("No-label AQI exclusion and daily categories follow Notebooks/aqi_labels.py.")

if __name__=="__main__": main()


