import streamlit as st

def app():
    st.title("🏃 Athlete Race Info Form")

    # Initialize session_state variables
    if 'athlete_name' not in st.session_state:
        st.session_state['athlete_name'] = ""
    if 'race_name' not in st.session_state:
        st.session_state['race_name'] = ""
    if 'race_date' not in st.session_state:
        st.session_state['race_date'] = None
    if 'kilometers' not in st.session_state or st.session_state['kilometers'] < 0.1:
        st.session_state['kilometers'] = 0.1
    if 'form_submitted' not in st.session_state:
        st.session_state['form_submitted'] = False

    # --- Form ---
    with st.form("race_info_form"):
        athlete_name = st.text_input("🏃 Athlete's Name", st.session_state['athlete_name'])
        race_name = st.text_input("🏁 Race to be Analyzed", st.session_state['race_name'])
        race_date = st.date_input("📅 Date of the Race", st.session_state['race_date'])
        kilometers = st.number_input(
            "📏 Kilometers Run", min_value=0.1, step=0.1, value=st.session_state['kilometers']
        )

        # --- Submit button inside the form ---
        submitted = st.form_submit_button("Submit Form")
        if submitted:
            st.session_state['athlete_name'] = athlete_name
            st.session_state['race_name'] = race_name
            st.session_state['race_date'] = race_date
            st.session_state['kilometers'] = kilometers
            st.session_state['form_submitted'] = True
            st.success("✅ Form submitted! Now go to the Analysis page to upload your .fit file and see results.")

        # --- Button to go to Analysis ---
    if st.session_state.get('form_submitted', False):
        if st.button("Go to Analysis"):
            st.session_state['page'] = 'analysis'        