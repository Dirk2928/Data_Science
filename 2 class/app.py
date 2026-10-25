import streamlit as st
prediction=st.Page("pages/prediction.py",title="Next-day Clean vs Polluted",default=True)
results=st.Page("pages/results.py",title="Model results")
database=st.Page("pages/database.py",title="Daily database data")
st.navigation([prediction,results,database],position="sidebar").run()
