import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px

st.set_page_config(page_title="DU COACHING RACE Analyzer", layout="wide")
st.title("📊 DU COACHING RACE Analyzer")
st.info("This analyzer is brought to you by coach Davide Ambrosini")

# --- Athlete and race info form ---
with st.form("race_info_form"):
    athlete_name = st.text_input("🏃 Athlete's Name")
    race_name = st.text_input("🏁 Race to be Analyzed")
    race_date = st.date_input("📅 Date of the Race")
    kilometers = st.number_input("📏 Kilometers Run", min_value=0.1, step=0.1)
    submitted = st.form_submit_button("Submit Info")

# Green flag if form is fully filled and submitted
if submitted and athlete_name and race_name and kilometers:
    st.success("✅ Form submitted successfully!")

# File uploader
uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

if uploaded_file is not None:
    if not uploaded_file.name.endswith(".fit"):
        st.error("❌ The uploaded file is not a .fit file.")
    else:
        try:
            # Read FIT file
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
                start_time = df["time"].iloc[0]
                df["elapsed_sec"] = (df["time"] - start_time).dt.total_seconds()
                df["elapsed_hours"] = df["elapsed_sec"] / 3600
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

                total_seconds = df["elapsed_sec"].iloc[-1]
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)
                final_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
                st.markdown(f"**Final Time:** {final_time_str}")
                st.markdown("---")

                # Smooth HR
                df["hr_smooth"] = df["hr"].rolling(window=3, min_periods=1).mean()

                # HR averages
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

                # Linear regression for trend line
                X = df["elapsed_sec"].values.reshape(-1, 1)
                y = df["hr_smooth"].values
                reg = LinearRegression().fit(X, y)
                slope_m = abs(reg.coef_[0])
                det_index = slope_m * 10000

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

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")
                tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\n"
                                "DI = 7 - DECADIMENTO MEDIO\n"
                                "DI > 10 - ALTO DECADIMENTO")
                st.markdown(
                    f"<div title='{tooltip_text}' style='font-size:16px; background-color:{color}; color:black; padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", 
                    unsafe_allow_html=True
                )

                # --- Interactive Plotly chart for Streamlit ---
                df["trend_line"] = reg.predict(X)

                # Hover text with [h]:mm | HR
                df["hover_text"] = df.apply(
                    lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm",
                    axis=1
                )

                fig = px.line(
                    df,
                    x="elapsed_hours",
                    y="hr_smooth",
                    labels={"elapsed_hours": "Elapsed Time (hours)", "hr_smooth": "Heart Rate (bpm)"},
                    title="Heart Rate Over Time",
                    hover_data={"hover_text": True, "elapsed_hours": False, "hr_smooth": False}
                )

                # Trend line
                fig.add_scatter(
                    x=df["elapsed_hours"],
                    y=df["trend_line"],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                )

                fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))
                fig.update_yaxes(tickformat='d')  # Y-axis integer
                st.plotly_chart(fig, use_container_width=True)

                # --- PDF generation with Matplotlib chart ---
                if st.button("📄 Generate PDF Report"):
                    pdf = FPDF()
                    pdf.add_page()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    pdf.set_font("Arial", "B", 16)
                    pdf.cell(0, 10, "DU COACHING RACE Analyzer Report", ln=True, align="C")
                    pdf.ln(10)

                    pdf.set_font("Arial", "", 12)
                    pdf.cell(0, 8, f"Athlete: {athlete_name}", ln=True)
                    pdf.cell(0, 8, f"Race: {race_name}", ln=True)
                    pdf.cell(0, 8, f"Date: {formatted_date}", ln=True)
                    pdf.cell(0, 8, f"Distance: {kilometers} km", ln=True)
                    pdf.cell(0, 8, f"Final Time: {final_time_str}", ln=True)
                    pdf.ln(5)
                    pdf.cell(0, 8, f"Overall Average HR: {overall_avg:.0f} bpm", ln=True)
                    pdf.cell(0, 8, f"First Half Avg HR: {first_half_avg:.0f} bpm", ln=True)
                    pdf.cell(0, 8, f"Second Half Avg HR: {second_half_avg:.0f} bpm", ln=True)
                    pdf.cell(0, 8, f"% Difference: {percent_diff:.1f}%", ln=True)
                    pdf.cell(0, 8, f"DET Index: {det_index_str} ({comment})", ln=True)
                    pdf.ln(10)

                    # Matplotlib chart
                    plt.figure(figsize=(10, 4))
                    plt.plot(df["elapsed_hours"], df["hr_smooth"], label="HR Smooth", color="blue")
                    plt.plot(df["elapsed_hours"], reg.predict(X), label="Trend Line", color="red", linestyle="--")
                    plt.xlabel("Elapsed Time (hours)")
                    plt.ylabel("Heart Rate (bpm)")
                    plt.title("Heart Rate Over Time")
                    plt.legend()
                    plt.tight_layout()

                    # X-axis in hours only
                    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
                    plt.yticks(np.arange(int(df["hr_smooth"].min()), int(df["hr_smooth"].max())+1, 5))

                    chart_buf = io.BytesIO()
                    plt.savefig(chart_buf, format="PNG")
                    chart_buf.seek(0)
                    pdf.image(chart_buf, x=10, w=190)

                    pdf_buffer = io.BytesIO()
                    pdf.output(pdf_buffer)
                    pdf_buffer.seek(0)

                    st.download_button(
                        label="⬇️ Download PDF",
                        data=pdf_buffer,
                        file_name=f"{athlete_name}_{race_name}_report.pdf",
                        mime="application/pdf"
                    )

        except Exception as e:
            st.error(f"❌ Error reading FIT file: {e}")
else:
    st.info("👆 Please upload a .fit file to begin.")
