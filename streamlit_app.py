import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.title("📊 DU COACHING RACE Analyzer")
st.info("This analyzer is brought to you by coach Davide Ambrosini")

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
                df["elapsed_str"] = pd.to_timedelta(df["elapsed_sec"], unit='s').apply(lambda x: str(x).split(".")[0])

                # Light smoothing
                df["hr_smooth"] = df["hr"].rolling(window=3, min_periods=1).mean()

                # Average HR calculations
                mid_index = len(df) // 2
                first_half_avg = df["hr"][:mid_index].mean()
                second_half_avg = df["hr"][mid_index:].mean()
                overall_avg = df["hr"].mean()
                percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

                # Display averages
                st.success(f"✅ Overall Average HR: **{overall_avg:.1f} bpm**")
                st.info(f"🟢 First Half Average: **{first_half_avg:.1f} bpm**")
                st.info(f"🔵 Second Half Average: **{second_half_avg:.1f} bpm**")

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
                det_index_str = f"{det_index:.2f}"
                if det_index < 4:
                    comment = "Scarso decadimento"
                    color = "green"
                elif det_index <= 10:
                    comment = "Decadimento medio"
                    color = "orange"
                else:
                    comment = "Alto decadimento"
                    color = "red"

                # Spacing before DET INDEX
                st.markdown("<br>", unsafe_allow_html=True)

                # Fixed explanation line
                st.markdown("**IL DET INDEX INDICA IL DECADIMENTO DELLA FC NEL CORSO DEL TEMPO.**")

                # Tooltip explanation
                tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\n"
                                "DI = 7 - DECADIMENTO MEDIO\n"
                                "DI > 10 - ALTO DECADIMENTO")
                st.markdown(
                    f"<div title='{tooltip_text}' style='font-size:16px; color:{color}'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", 
                    unsafe_allow_html=True
                )

                # Create trend line values
                df["trend_line"] = reg.predict(X)

                # Plot interactive line chart with trend line
                fig = px.line(
                    df,
                    x="elapsed_hours",
                    y="hr_smooth",
                    labels={"elapsed_hours": "Elapsed Time (hours)", "hr_smooth": "Heart Rate (bpm)"},
                    title="Heart Rate Over Time",
                    hover_data={"elapsed_str": True, "hr_smooth": True, "elapsed_hours": False}
                )
                # Add trend line as a dashed red line
                fig.add_scatter(x=df["elapsed_hours"], y=df["trend_line"], mode='lines', 
                                line=dict(color='red', dash='dash'), name='Trend Line')

                fig.update_traces(mode='lines', line=dict(color='blue'), selector=dict(name='hr_smooth'))
                fig.update_layout(xaxis_title="Elapsed Time (hours)", xaxis_rangeslider_visible=True)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading FIT file: {e}")

else:
    st.info("👆 Please upload a .fit file to begin.")
