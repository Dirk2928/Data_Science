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
        if "aqi_daily" in tables:
            source="aqi_daily"; stations=pd.read_sql_query(text("SELECT DISTINCT station FROM aqi_daily ORDER BY station"),con).station.dropna().tolist()
        elif "all_regions" in tables:
            source="all_regions"; stations=pd.read_sql_query(text("SELECT DISTINCT station FROM all_regions ORDER BY station"),con).station.dropna().tolist()
        else:
            source=[x for x in STATIONS if x in tables]
            if not source: raise RuntimeError("No aqi_daily, all_regions, or station tables found.")
            stations=source
    choice=st.selectbox("Station",["All stations",*stations])
    with engine.connect() as con:
        if isinstance(source,str):
            count_query=f"SELECT COUNT(*) FROM `{source}`" if choice=="All stations" else f"SELECT COUNT(*) FROM `{source}` WHERE station=:station"
            count_params={} if choice=="All stations" else {"station":choice}
            total_rows=con.execute(text(count_query),count_params).scalar_one()
        else:
            choices=source if choice=="All stations" else [choice]
            table_counts={
                table:con.execute(text(f"SELECT COUNT(*) FROM `{table}`")).scalar_one()
                for table in choices
            }
            total_rows=sum(table_counts.values())
        total_pages=(total_rows+49)//50
        page=st.number_input("Page",min_value=1,max_value=max(1,total_pages),value=1,step=1)
        st.caption(f"Page {int(page)} of {total_pages} · {total_rows:,} rows total (50 rows per page)")
        offset=(int(page)-1)*50
        if isinstance(source,str):
            q=f"SELECT * FROM `{source}` LIMIT :lim OFFSET :off" if choice=="All stations" else f"SELECT * FROM `{source}` WHERE station=:station LIMIT :lim OFFSET :off"
            params={"lim":50,"off":offset}
            if choice!="All stations": params["station"]=choice
            frame=pd.read_sql_query(text(q),con,params=params)
        else:
            parts=[]; remaining_offset=offset; remaining_rows=50
            for table in choices:
                table_count=table_counts[table]
                if remaining_offset >= table_count:
                    remaining_offset-=table_count
                    continue
                one=pd.read_sql_query(
                    text(f"SELECT * FROM `{table}` LIMIT :lim OFFSET :off"),
                    con,
                    params={"lim":remaining_rows,"off":remaining_offset},
                )
                if "station" not in one: one["station"]=table
                parts.append(one)
                remaining_rows-=len(one)
                remaining_offset=0
                if remaining_rows==0: break
            frame=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    st.write(f"Source: {source if isinstance(source,str) else 'station tables'}")
    st.dataframe(frame,use_container_width=True)
except Exception as e:
    st.error("Unable to read MySQL air_quality. Run air-quality-classification/notebooks/DB_Connection.ipynb and etl.ipynb first, then check DB_URL.")
    st.caption(str(e))
finally:
    if engine is not None: engine.dispose()
