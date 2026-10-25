import os
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine,text
st.title("Daily database data")
st.caption("Read-only sample from air_quality.aqi_daily.")
engine=None
try:
    engine=create_engine(os.environ.get("DB_URL","mysql+pymysql://root:@localhost/air_quality"),pool_pre_ping=True)
    with engine.connect() as connection:
        total=int(pd.read_sql_query(text("SELECT COUNT(*) AS n FROM aqi_daily"),connection).iloc[0,0])
        stations=pd.read_sql_query(text("SELECT DISTINCT station FROM aqi_daily ORDER BY station"),connection).station.dropna().tolist()
    choice=st.selectbox("Station",["All stations",*stations]); page=st.number_input("Page",min_value=1,value=1,step=1); offset=(int(page)-1)*50
    query="SELECT * FROM aqi_daily ORDER BY date,station LIMIT :lim OFFSET :off" if choice=="All stations" else "SELECT * FROM aqi_daily WHERE station=:station ORDER BY date LIMIT :lim OFFSET :off"
    params={"lim":50,"off":offset} if choice=="All stations" else {"station":choice,"lim":50,"off":offset}
    with engine.connect() as connection: frame=pd.read_sql_query(text(query),connection,params=params)
    st.write(f"Rows in table: {total:,}. Showing up to 50 rows."); st.dataframe(frame,use_container_width=True)
except Exception as error:
    st.error("Could not read air_quality.aqi_daily. Check that MySQL is running and DB_URL is correct."); st.caption(str(error))
finally:
    if engine is not None: engine.dispose()
