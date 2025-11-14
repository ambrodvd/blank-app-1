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
    athlete_name = st.text_input("🏃 Athlete's Name", value=st.session_state.get('athlete_name', ''))
    race_name = st.text_input("🏁 Race to be Analyzed", value=st.session_state.get('race_name', ''))
    race_date = st.date_input("📅 Date of the Race", value=st.session_state.get('race_date'))
    kilometers = st.number_input("📏 Kilometers Run", min_value=0.1, step=0.1, value=st.session_state.get('kilometers', 5.0))
    info_submitted = st.form_submit_button("Submit Info")

if info_submitted:
    st.session_state['athlete_name'] = athlete_name
    st.session_state['race_name'] = race_name
    st.session_state['race_date'] = race_date
    st.session_state['kilometers'] = kilometers
    st.success("✅ Form submitted successfully!")

# --- Heart Rate Zones Form ---
with st.form("hr_zones_form"):
    st.subheader("❤️ Athlete Heart Rate Zones")
    st.caption("Please input the *upper limit (in bpm)* for each training zone:")

    z1 = st.number_input("Zone 1 (Recovery) – up to:", min_value=60, step=1, value=st.session_state.get('z1', 140))
    z2 = st.number_input("Zone 2 (Aerobic) – up to:", min_value=60, step=1, value=st.session_state.get('z2', 160))
    z3 = st.number_input("Zone 3 (Tempo) – up to:", min_value=60, step=1, value=st.session_state.get('z3', 170))
    z4 = st.number_input("Zone 4 (Sub Threshold) – up to:", min_value=60, step=1, value=st.session_state.get('z4', 180))
    z5 = st.number_input("Zone 5 (Super Threshold) – up to:", min_value=60, step=1, value=st.session_state.get('z5', 200))

    zones_submitted = st.form_submit_button("Submit HR Zones")

if zones_submitted:
    if not (z1 < z2 < z3 < z4 < z5):
        st.error("⚠️ There's something wrong in the HR data. Please correct the values.")
    else:
        st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5'] = z1, z2, z3, z4, z5
        st.success("✅ Heart Rate Zones saved successfully!")
        st.write(f"""
        **HR Zones:**
        - 🩵 Zone 1 (Recovery): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic): {z1+1}–{z2} bpm  
        - 💛 Zone 3 (Tempo): {z2+1}–{z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1}–{z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1}–{z5} bpm
        """)

# --- Time Segment Input Form (Start → End) ---
with st.form("time_segment_form"):
    st.subheader("⏱️ Time segments for Partial Analysis")
    st.caption("Please choose your time segment (H:MM) for the time-in-zone analysis")

    col1, col2 = st.columns(2)
    with col1:
        segment1_start = st.text_input("Segment 1 Start", value=st.session_state.get('segment1_start', "0:00"))
        segment2_start = st.text_input("Segment 2 Start", value=st.session_state.get('segment2_start', "0:00"))
        segment3_start = st.text_input("Segment 3 Start", value=st.session_state.get('segment3_start', "0:00"))

    with col2:
        segment1_end = st.text_input("Segment 1 End", value=st.session_state.get('segment1_end', "1:00"))
        segment2_end = st.text_input("Segment 2 End", value=st.session_state.get('segment2_end', "2:00"))
        segment3_end = st.text_input("Segment 3 End", value=st.session_state.get('segment3_end', "3:00"))

    segments_submitted = st.form_submit_button("Save Time Segments")

if segments_submitted:
    st.session_state['segment1_start'], st.session_state['segment1_end'] = segment1_start, segment1_end
    st.session_state['segment2_start'], st.session_state['segment2_end'] = segment2_start, segment2_end
    st.session_state['segment3_start'], st.session_state['segment3_end'] = segment3_start, segment3_end
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
                if 'athlete_name' not in st.session_state:
                    st.warning("⚠️ Please submit the Athlete and Race info in the form above")
                else:
                    st.markdown("---")
                    st.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
                    st.markdown(f"**Race:** {st.session_state['race_name']}")
                    formatted_date = st.session_state['race_date'].strftime("%d/%m/%Y")
                    st.markdown(f"**Date:** {formatted_date}")
                    st.markdown(f"**Distance:** {st.session_state['kilometers']} km")

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
                if all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):
                    z1, z2, z3, z4, z5 = st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5']

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
                    zone_summary["Time [h:mm]"] = zone_summary["time_diff_sec"].apply(lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}")
                    zone_order = ["Zone 1 – Recovery","Zone 2 – Aerobic","Zone 3 – Tempo","Zone 4 – Sub Threshold","Zone 5 – Super Threshold"]
                    zone_summary["order"] = zone_summary["HR Zone"].apply(lambda z: zone_order.index(z))
                    zone_summary = zone_summary.sort_values("order")
                    st.markdown("### ⏱️ Time Spent in Each HR Zone")
                    st.dataframe(zone_summary[["HR Zone", "Time [h:mm]"]].set_index("HR Zone"))

                                    # --- Combined Segment Table with Total ---
                if all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):
                    z1, z2, z3, z4, z5 = st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5']

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

                    zone_order = ["Zone 1 – Recovery","Zone 2 – Aerobic","Zone 3 – Tempo","Zone 4 – Sub Threshold","Zone 5 – Super Threshold"]

                    # Total (overall) time-in-zone
                    total_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)

                    # Prepare segment inputs
                    if all(k in st.session_state for k in ['segment1_start','segment1_end','segment2_start','segment2_end','segment3_start','segment3_end']):
                        segment_inputs = [
                            (st.session_state['segment1_start'], st.session_state['segment1_end'], "Segment 1"),
                            (st.session_state['segment2_start'], st.session_state['segment2_end'], "Segment 2"),
                            (st.session_state['segment3_start'], st.session_state['segment3_end'], "Segment 3"),
                        ]

                        segment_data = {}

                        def h_mm_to_seconds(hmm):
                            try:
                                h, m = hmm.split(":")
                                return int(h)*3600 + int(m)*60
                            except:
                                return None

                        for start_str, end_str, label in segment_inputs:
                            start_sec = h_mm_to_seconds(start_str)
                            end_sec = h_mm_to_seconds(end_str)
                            if start_sec is None or end_sec is None or start_sec >= end_sec:
                                segment_data[label] = pd.Series(0, index=zone_order)
                                continue

                            df_segment = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()
                            df_segment["time_diff_sec"] = df_segment["elapsed_sec"].diff().fillna(0)
                            seg_summary = df_segment.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
                            segment_data[label] = seg_summary

                        # Combine all segments + total into one DataFrame
                        combined_df = pd.DataFrame(segment_data)
                        combined_df["Total"] = total_summary
                        # Format as H:MM
                        combined_df = combined_df.applymap(lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}")

                        st.markdown("### ⏱️ Time-in-Zone Table (Segments + Total)")
                        st.dataframe(combined_df)


                # --- DET Index ---
                X = df["elapsed_sec"].values.reshape(-1,1)
                y = df["hr_smooth"].values
                reg = LinearRegression().fit(X,y)
                slope_m = abs(reg.coef_[0])
                det_index = slope_m * 10000
                det_index_str = f"{det_index:.1f}"
                if det_index <4: comment,color="Scarso decadimento","green"
                elif det_index<=10: comment,color="Decadimento medio","cyan"
                else: comment,color="Alto decadimento","lightcoral"

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")
                tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\nDI = 7 - DECADIMENTO MEDIO\nDI > 10 - ALTO DECADIMENTO")
                st.markdown(f"<div title='{tooltip_text}' style='font-size:16px; background-color:{color}; color:black; padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", unsafe_allow_html=True)

                # --- Plotly chart ---
                df["trend_line"] = reg.predict(X)
                df["Race Time [h:mm] "] = df.apply(lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm", axis=1)

                fig = px.line(df, x="elapsed_hours", y="hr_smooth",
                              labels={"elapsed_hours":"Elapsed Time (hours)", "hr_smooth":"Heart Rate (bpm)"},
                              title="Heart Rate Over Time",
                              hover_data={"Race Time [h:mm] ":True,"elapsed_hours":False,"hr_smooth":False})
                fig.add_scatter(x=df["elapsed_hours"], y=df["trend_line"], mode='lines', line=dict(color='red', dash='dash'), name='Trend Line')
                fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))
                fig.update_yaxes(tickformat='d')
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error reading FIT file: {e}")

else:
    st.info("👆 Please upload a .fit file to begin.")