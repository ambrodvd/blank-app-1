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
    info_submitted = st.form_submit_button("Submit Info")

if info_submitted and athlete_name and race_name and kilometers:
    st.success("✅ Form submitted successfully!")

# --- Heart Rate Zones Form ---
with st.form("hr_zones_form"):
    st.subheader("❤️ Athlete Heart Rate Zones")
    st.caption("Please input the *upper limit (in bpm)* for each training zone:")

    z1 = st.number_input("Zone 1 (Recovery) – up to:", min_value=60, step=1)
    z2 = st.number_input("Zone 2 (Aerobic) – up to:", min_value=60, step=1)
    z3 = st.number_input("Zone 3 (Tempo) – up to:", min_value=60, step=1)
    z4 = st.number_input("Zone 4 (Sub Threshold) – up to:", min_value=60, step=1)
    z5 = st.number_input("Zone 5 (Super Threshold) – up to:", min_value=60, step=1, value=255)

    zones_submitted = st.form_submit_button("Submit HR Zones")

if zones_submitted:
    if not (z1 < z2 < z3 < z4 < z5):
        st.error("⚠️ There's something wrong in the HR data. Please correct the values.")
    else:
        st.success("✅ Heart Rate Zones saved successfully!")
        st.write(f"""
        **HR Zones:**
        - 🩵 Zone 1 (Recovery): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic): {z1+1}–{z2} bpm  
        - 💛 Zone 3 (Tempo): {z2+1}–{z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1}–{z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1}–{z5} bpm
        """)

# --- Time segment Input Form (Start → End) ---
with st.form("time_segment_form"):
    st.subheader("⏱️ Time segment for Partial Analysis")
    st.caption("Please choose your time segment (h:mm) for the time in zone analysis")

    col1, col2 = st.columns(2)
    with col1:
        segment1_start = st.text_input("Segment 1 Start", value="0:00")
        segment2_start = st.text_input("Segment 2 Start", value="0:00")
        segment3_start = st.text_input("Segment 3 Start", value="0:00")

    with col2:
        segment1_end = st.text_input("Segment 1 End", value="1:00")
        segment2_end = st.text_input("Segment 2 End", value="2:00")
        segment3_end = st.text_input("Segment 3 End", value="3:00")

    segments_submitted = st.form_submit_button("Save Time Segments")

if segments_submitted:
    st.success("✅ Time segments saved successfully!")

# --- FIT file uploader ---
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
                df = pd.DataFrame(hr_data)
                start_time = df["time"].iloc[0]
                df["elapsed_sec"] = (df["time"] - start_time).dt.total_seconds()
                df["elapsed_hours"] = df["elapsed_sec"] / 3600
                df["elapsed_hms"] = df["elapsed_sec"].apply(
                    lambda x: f"{int(x // 3600)}:{int((x % 3600) // 60):02d}:{int(x % 60):02d}"
                )

                # --- Athlete & race info display ---
                if info_submitted == False:
                    st.warning("⚠️ Please submit the Athlete and Race info in the form above")

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

                # --- Smooth HR ---
                df["hr_smooth"] = df["hr"].rolling(window=3, min_periods=1).mean()

                # --- HR averages ---
                mid_index = len(df) // 2
                first_half_avg = df["hr"][:mid_index].mean()
                second_half_avg = df["hr"][mid_index:].mean()
                overall_avg = df["hr"].mean()
                percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

                st.markdown(f"❤️ Overall Average HR: **{overall_avg:.0f} bpm**")
                st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
                st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")
                if percent_diff >= -10:
                    st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
                else:
                    st.error(f"📊 % Difference: **{percent_diff:.1f}%**")

                                # --- HR Zones / Time in Zone ---
                if zones_submitted == False:
                    st.warning("⚠️ Please submit the Heart Rate Zones in the form above to see time spent in each zone.")
                zone_summary = None
                if uploaded_file is not None:
                    def get_hr_zone(hr):
                        if hr <= z1:
                            return "Zone 1 – Recovery"
                        elif hr <= z2:
                            return "Zone 2 – Aerobic"
                        elif hr <= z3:
                            return "Zone 3 – Tempo"
                        elif hr <= z4:
                            return "Zone 4 – Sub Threshold"
                        else:
                            return "Zone 5 – Super Threshold"

                    df["HR Zone"] = df["hr"].apply(get_hr_zone)
                    df["time_diff_sec"] = df["elapsed_sec"].diff().fillna(0)
                    zone_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reset_index()
                    zone_summary["Time [h:mm]"] = zone_summary["time_diff_sec"].apply(
                        lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}"
                    )
                    zone_order = [
                        "Zone 1 – Recovery",
                        "Zone 2 – Aerobic",
                        "Zone 3 – Tempo",
                        "Zone 4 – Sub Threshold",
                        "Zone 5 – Super Threshold"
                    ]
                    zone_summary["order"] = zone_summary["HR Zone"].apply(lambda z: zone_order.index(z))
                    zone_summary = zone_summary.sort_values("order")
                    st.markdown("### ⏱️ Time Spent in Each HR Zone")
                    st.dataframe(zone_summary[["HR Zone", "Time [h:mm]"]].set_index("HR Zone"))

                # --- TIME IN ZONE FOR CUSTOM SEGMENTS ---
                
                if zones_submitted == False:
                    st.warning("⚠️ Please submit the Heart Rate Zones in the form above to see time spent in each zone.")
                if segments_submitted == False:
                    st.warning("⚠️ Please submit the time segments in the form above to see time spent in each zone.")

                if segments_submitted and zones_submitted:

                    st.markdown("## 🔍 Time-in-Zone for Custom Segments")
                    st.caption("Each segment uses its own start and end time. Format: H:MM.")

                    def h_mm_to_seconds(hmm):
                        """Convert H:MM to total seconds."""
                        try:
                            h, m = hmm.split(":")
                            return int(h) * 3600 + int(m) * 60
                        except:
                            return None

                    # Convert segment inputs
                    segment_inputs = [
                        (segment1_start, segment1_end, "Segment 1"),
                        (segment2_start, segment2_end, "Segment 2"),
                        (segment3_start, segment3_end, "Segment 3"),
                    ]

                    zone_order = [
                        "Zone 1 – Recovery",
                        "Zone 2 – Aerobic",
                        "Zone 3 – Tempo",
                        "Zone 4 – Sub Threshold",
                        "Zone 5 – Super Threshold"
                    ]

                    for start_str, end_str, label in segment_inputs:
                        start_sec = h_mm_to_seconds(start_str)
                        end_sec = h_mm_to_seconds(end_str)

                        if start_sec is None or end_sec is None:
                            st.error(f"❌ Invalid time format in {label}. Use H:MM.")
                            continue

                        if start_sec >= end_sec:
                            st.error(f"⚠️ {label} start time must be *before* end time.")
                            continue

                        # Extract segment dataframe
                        df_segment = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()

                        if df_segment.empty:
                            st.warning(f"⚠️ {label} contains no data in the selected time range.")
                            continue

                        # Compute time differences
                        df_segment["time_diff_sec"] = df_segment["elapsed_sec"].diff().fillna(0)

                        # Summarize by zone
                        zone_summary_segment = df_segment.groupby("HR Zone")["time_diff_sec"].sum().reset_index()
                        zone_summary_segment["Time [h:mm]"] = zone_summary_segment["time_diff_sec"].apply(
                            lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}"
                        )

                        # Sort by zone order
                        zone_summary_segment["order"] = zone_summary_segment["HR Zone"].apply(lambda z: zone_order.index(z))
                        zone_summary_segment = zone_summary_segment.sort_values("order")

                        # Display table
                        st.markdown(f"### ⏱️ {label}: {start_str} → {end_str}")
                        st.dataframe(zone_summary_segment[["HR Zone", "Time [h:mm]"]].set_index("HR Zone"))
                        st.markdown("---")


                # --- Linear Regression & DET Index ---
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

                # --- Plotly chart ---
                df["trend_line"] = reg.predict(X)
                df["Race Time [h:mm] "] = df.apply(
                    lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm",
                    axis=1
                )

                fig = px.line(
                    df,
                    x="elapsed_hours",
                    y="hr_smooth",
                    labels={"elapsed_hours": "Elapsed Time (hours)", "hr_smooth": "Heart Rate (bpm)"},
                    title="Heart Rate Over Time",
                    hover_data={"Race Time [h:mm] ": True, "elapsed_hours": False, "hr_smooth": False}
                )

                fig.add_scatter(
                    x=df["elapsed_hours"],
                    y=df["trend_line"],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Trend Line'
                )
                fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))
                fig.update_yaxes(tickformat='d')
                st.plotly_chart(fig, use_container_width=True)

                # --- PDF Generation ---
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
                    pdf.ln(5)
                    if zone_summary is not None and not zone_summary.empty:
                        pdf.set_font("Arial", "B", 12)
                        pdf.cell(0, 8, "Time Spent in Each HR Zone", ln=True)
                        pdf.ln(3)
                        pdf.set_font("Arial", "", 12)
                        for _, row in zone_summary.iterrows():  # <-- you need this loop
                            # Replace Unicode dash to avoid FPDF errors
                            zone_name_pdf = row['HR Zone'].replace("–", "-")
                            pdf.cell(0, 8, f"{zone_name_pdf}: {row['Time [h:mm]']}", ln=True)
                        pdf.ln(5)


                    plt.figure(figsize=(10, 4))
                    plt.plot(df["elapsed_hours"], df["hr_smooth"], label="HR Smooth", color="blue")
                    plt.plot(df["elapsed_hours"], reg.predict(X), label="Trend Line", color="red", linestyle="--")
                    plt.xlabel("Elapsed Time (hours)")
                    plt.ylabel("Heart Rate (bpm)")
                    plt.title("Heart Rate Over Time")
                    plt.legend()
                    plt.tight_layout()
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
