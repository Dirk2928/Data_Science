import streamlit as st
prediction=st.Page("pages/prediction.py",title="Next-day 3-class prediction",default=True)
results=st.Page("pages/results.py",title="Model results")
database=st.Page("pages/database.py",title="Database data")
st.navigation([prediction,results,database],position="sidebar").run()
