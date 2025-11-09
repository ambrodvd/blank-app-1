import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 DU COACHING RACE Analyzer")
st.info("This analyzer is brought to you by coach Davide Ambrosini")

# --- Athlete and race info ---
with st.form("race_info_form"):
    athlete_name = st.text_input("🏃 Athlete's Name")
    race_name = st.text_input("🏁 Race to be Analyzed")
    race_date = st.date_input("📅 Date of the Race")
    kilometers = st.number_input("📏 Kilometers Run", min_value=0.1, step=0.1)

    submitted = st.form_submit_button("Submit Info")

# File uploader
uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

if uploaded_file is not None:
    if not uploaded_file.name.endswith(".fit"):
        st.error("❌ The uploaded file is not a .fit file.")
    else:
        try:
            # Read the FIT file
            fitfile = FitFile(io.BytesIO(uploaded_file.getvalue()))
            
            # Extract HR values and timestamps
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
            else:
                # Create DataFrame
                df = pd.DataFrame(hr_data)

                # Reset time to start at 0
                start_time = df["time"].iloc[0]
                df["elapsed_sec"] = (df["time"] - start_time).dt.total_seconds()
                df["elapsed_hours"] = df["elapsed_sec"] / 3600

                # Create elapsed time as [h:mm:ss] without days
                df["elapsed_hms"] = df["elapsed_sec"].apply(
                    lambda x: f"{int(x // 3600)}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
                )

                # --- Display athlete and race info ---
                st.markdown("---")
                st.markdown(f"**Athlete:** {athlete_name}")
                st.markdown(f"**Race:** {race_name}")
                formatted_date = race_date.strftime("%d/%m/%Y")
                st.markdown(f"**Date:** {formatted_date}")
                st.markdown(f"**Distance:** {kilometers} km")

                # --- Calculate final time from FIT data ---
                total_seconds = df["elapsed_sec"].iloc[-1]
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                final_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"

                # Display final time
                st.markdown(f"**Final Time:** {final_time_str}")
                st.markdown("---")


                # Light smoothing
                df["hr_smooth"] = df["hr"].rolling(window=3, min_periods=1).mean()

                # Average HR calculations
                mid_index = len(df) // 2
                first_half_avg = df["hr"][:mid_index].mean()
                second_half_avg = df["hr"][mid_index:].mean()
                overall_avg = df["hr"].mean()
                percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

                # Display averages
                st.success(f"✅ Overall Average HR: **{overall_avg:.0f} bpm**")
                st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
                st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")

                # Colored box for % difference
                if percent_diff >= -10:
                    st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
                else:
                    st.error(f"📊 % Difference: **{percent_diff:.1f}%**")

                # Linear regression for trend line
                X = df["elapsed_sec"].values.reshape(-1, 1)  # seconds as predictor
                y = df["hr_smooth"].values
                reg = LinearRegression().fit(X, y)
                slope_m = abs(reg.coef_[0])  # always positive
                det_index = slope_m * 10000

                # DET INDEX formatting and comment with color
                det_index_str = f"{det_index:.1f}"
                if det_index < 4:
                    comment = "Scarso decadimento"
                    color = "green"
                elif det_index <= 10:
                    comment = "Decadimento medio"
                    color = "cyan"
                else:
                    comment = "Alto decadimento"
                    color = "lightcoral"

                # Spacing before DET INDEX
                st.markdown("<br>", unsafe_allow_html=True)

                # Fixed explanation line
                st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")

                # Tooltip explanation
                tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\n"
                                "DI = 7 - DECADIMENTO MEDIO\n"
                                "DI > 10 - ALTO DECADIMENTO")
                st.markdown(
                    f"<div title='{tooltip_text}' style='font-size:16px; ; background-color:{color}; color:black'padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", 
                    unsafe_allow_html=True
                )

                # Create trend line values
                df["trend_line"] = reg.predict(X)

                # Create a custom hover text
                df["Elapsed time:"] = df.apply(
                    lambda row: f"{row['elapsed_hms']} | HR: {row['hr_smooth']:.0f} bpm", axis=1
                )

                # Plot interactive line chart with trend line
                fig = px.line(
                    df,
                    x="elapsed_hours",
                    y="hr_smooth",
                    labels={"elapsed_hours": "Elapsed Time (hours)", "hr_smooth": "Heart Rate (bpm)"},
                    title="Heart Rate Over Time",
                    hover_data={"Elapsed time:": True, "elapsed_hours": False, "hr_smooth": False}  # hide default
                )

                # Add trend line as a dashed red line
                fig.add_scatter(
                    x=df["elapsed_hours"],
                    y=df["trend_line"],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                )

                # Tell Plotly to use your custom text for hover
                fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))

                fig.update_layout(
                    xaxis_title="Elapsed Time (hours)",
                    xaxis_rangeslider_visible=True
                )

                st.plotly_chart(fig, use_container_width=True)


        except Exception as e:
            st.error(f"❌ Error reading FIT file: {e}")

else:
    st.info("👆 Please upload a .fit file to begin.")
