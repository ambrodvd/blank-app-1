import streamlit as st
from fitparse import FitFile
import io

def app():
    if 'form_submitted' not in st.session_state:
        st.session_state['form_submitted'] = False
    if 'uploaded_file' not in st.session_state:
        st.session_state['uploaded_file'] = None

    st.title("Athlete Race Data Upload")

    # Step 1: Form
    if not st.session_state['form_submitted']:
        with st.form("race_info_form"):
            st.session_state['athlete_name'] = st.text_input("🏃 Athlete's Name")
            st.session_state['race_name'] = st.text_input("🏁 Race to be Analyzed")
            st.session_state['race_date'] = st.date_input("📅 Date of the Race")
            st.session_state['kilometers'] = st.number_input("📏 Kilometers Run", min_value=0.1, step=0.1)
            submitted = st.form_submit_button("Submit Info")
            if submitted:
                st.session_state['form_submitted'] = True
                st.success("✅ Form submitted! Now upload the .fit file below.")

    # Step 2: File uploader
    if st.session_state['form_submitted']:
        uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])
        if uploaded_file is not None:
            st.session_state['uploaded_file'] = uploaded_file
            st.success(f"✅ File uploaded: {uploaded_file.name}")

            # --- Navigation button ---
            if st.button("Go to Analysis"):
                # set query parameter to simulate navigation
                st.session_state['page'] = 'analysis'
