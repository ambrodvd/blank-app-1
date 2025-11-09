import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

def app():
    st.title("📊 DU COACHING RACE Analyzer")
    st.info("This analyzer is brought to you by coach Davide Ambrosini")
    
    if st.button("Go to Upload"):
        st.session_state['page'] = "UPLOAD"

