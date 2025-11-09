# app.py
import streamlit as st
from Home import app as home_app
from Upload import app as upload_app
from Analysis import app as analysis_app

# Initialize session state
if 'page' not in st.session_state:
    st.session_state['page'] = "HOME"

# Page navigation
if st.session_state['page'] == "HOME":
    home_app()
elif st.session_state['page'] == "UPLOAD":
    upload_app()
elif st.session_state['page'] == "ANALYSIS":
    analysis_app()