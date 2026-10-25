import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine,inspect,text
STATIONS=["aotizhongxin","changping","dingling","dongsi","guanyuan","gucheng","huairou","nongzhanguan","shunyi","tiantan","wanliu","wanshouxigong"]
st.title("Database data")
st.caption("Live sample from MySQL air_quality; no CSV files are used.")
engine=None
try:
    engine=create_engine(os.environ.get("DB_URL","mysql+pymysql://root:@localhost/air_quality"))
    with engine.connect() as con:
        tables=set(inspect(con).get_table_names())
        if "all_regions" in tables:
            source="all_regions"; stations=pd.read_sql_query(text("SELECT DISTINCT station FROM all_regions ORDER BY station"),con).station.dropna().tolist()
        else:
            source=[x for x in STATIONS if x in tables]
            if not source: raise RuntimeError("No all_regions or station tables found.")
            stations=source
    choice=st.selectbox("Station",["All stations",*stations]); page=st.number_input("Page",min_value=1,value=1,step=1); offset=(int(page)-1)*50
    with engine.connect() as con:
        if isinstance(source,str):
            q="SELECT * FROM all_regions LIMIT :lim OFFSET :off" if choice=="All stations" else "SELECT * FROM all_regions WHERE station=:station LIMIT :lim OFFSET :off"
            params={"lim":50,"off":offset} if choice=="All stations" else {"station":choice,"lim":50,"off":offset}
            frame=pd.read_sql_query(text(q),con,params=params)
        else:
            choices=source if choice=="All stations" else [choice]; parts=[]
            for table in choices:
                one=pd.read_sql_query(text(f"SELECT * FROM `{table}` LIMIT 50 OFFSET {offset}"),con)
                if "station" not in one: one["station"]=table
                parts.append(one)
            frame=pd.concat(parts,ignore_index=True).head(50)
    st.write(f"Source: {source if isinstance(source,str) else 'station tables'} · 50 rows per page")
    st.dataframe(frame,use_container_width=True)
except Exception as e:
    st.error("Unable to read MySQL air_quality. Run air-quality-classification/notebooks/DB_Connection.ipynb and etl.ipynb first, then check DB_URL.")
    st.caption(str(e))
finally:
    if engine is not None: engine.dispose()
