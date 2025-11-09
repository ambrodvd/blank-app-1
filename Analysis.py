import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

def app():
    st.title("Race Analysis")

    # --- Check if the form and file were uploaded ---
    if 'form_submitted' not in st.session_state or not st.session_state['form_submitted']:
        st.warning("Please submit the race info in the Upload page first.")
        return

    if 'uploaded_file' not in st.session_state or st.session_state['uploaded_file'] is None:
        st.warning("Please upload a .fit file in the Upload page first.")
        return

    uploaded_file = st.session_state['uploaded_file']

    try:
        fitfile = FitFile(io.BytesIO(uploaded_file.getvalue()))
    except Exception as e:
        st.error(f"❌ Error reading FIT file: {e}")
        return

    # --- Extract HR data ---
    hr_data = []
    for record in fitfile.get_messages("record"):
        record_time = None
        hr = None
        for data in record:
            if data.name == "heart_rate":
                hr = data.value
            if data.name == "timestamp":
                record_time = data.value
        if hr is not None and record_time is not None:
            hr_data.append({"time": record_time, "hr": hr})

    if len(hr_data) == 0:
        st.warning("⚠️ No heart rate data found in this file.")
        return

    df = pd.DataFrame(hr_data)
    df["time"] = pd.to_datetime(df["time"])
    start_time = df["time"].iloc[0]
    df["elapsed_sec"] = (df["time"] - start_time).dt.total_seconds()
    df["elapsed_hours"] = df["elapsed_sec"] / 3600
    df["elapsed_hms"] = df["elapsed_sec"].apply(
        lambda x: f"{int(x // 3600)}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
    )

    # --- Display race info ---
    st.markdown("---")
    st.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
    st.markdown(f"**Race:** {st.session_state['race_name']}")
    st.markdown(f"**Date:** {st.session_state['race_date'].strftime('%d/%m/%Y')}")
    st.markdown(f"**Distance:** {st.session_state['kilometers']} km")

    # --- Final time ---
    total_seconds = df["elapsed_sec"].iloc[-1]
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    st.markdown(f"**Final Time:** {hours}:{minutes:02d}:{seconds:02d}")
    st.markdown("---")

    # --- Smooth HR ---
    df["hr_smooth"] = df["hr"].rolling(window=3, min_periods=1).mean()

    # --- HR averages ---
    mid_index = len(df) // 2
    first_half_avg = df["hr"][:mid_index].mean()
    second_half_avg = df["hr"][mid_index:].mean()
    overall_avg = df["hr"].mean()
    percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    st.success(f"✅ Overall Average HR: **{overall_avg:.0f} bpm**")
    st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
    st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")
    if percent_diff >= -10:
        st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
    else:
        st.error(f"📊 % Difference: **{percent_diff:.1f}%**")

    # --- Linear regression and trend line ---
    X = df["elapsed_sec"].values.reshape(-1, 1)
    y = df["hr_smooth"].values
    reg = LinearRegression().fit(X, y)
    df["trend_line"] = reg.predict(X)

    # --- DET index ---
    det_index = abs(reg.coef_[0]) * 10000
    if det_index < 4:
        comment = "Scarso decadimento"
        color = "green"
    elif det_index <= 10:
        comment = "Decadimento medio"
        color = "cyan"
    else:
        comment = "Alto decadimento"
        color = "lightcoral"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")
    tooltip_text = (
        "DI < 4 - SCARSO DECADIMENTO\n"
        "DI = 7 - DECADIMENTO MEDIO\n"
        "DI > 10 - ALTO DECADIMENTO"
    )
    st.markdown(
        f"<div title='{tooltip_text}' style='font-size:16px; background-color:{color}; color:black; padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index:.1f}</b> ({comment})</div>",
        unsafe_allow_html=True
    )

    # --- Plot HR over time ---
    df["Elapsed time:"] = df.apply(lambda row: f"{row['elapsed_hms']} | HR: {row['hr_smooth']:.0f} bpm", axis=1)
    fig = px.line(
        df,
        x="elapsed_hours",
        y="hr_smooth",
        labels={"elapsed_hours": "Elapsed Time (hours)", "hr_smooth": "Heart Rate (bpm)"},
        title="Heart Rate Over Time",
        hover_data={"Elapsed time:": True, "elapsed_hours": False, "hr_smooth": False}
    )
    fig.add_scatter(
        x=df["elapsed_hours"],
        y=df["trend_line"],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Trend Line'
    )
    fig.update_traces(customdata=df[["Elapsed time:"]], hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))
    fig.update_layout(xaxis_title="Elapsed Time (hours)", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)
