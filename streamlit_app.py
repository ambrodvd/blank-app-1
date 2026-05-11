import streamlit as st
from fitparse import FitFile
import io
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import math
import haversine
import tempfile

st.set_page_config(page_title="DU COACHING RACE Analyzer", layout="wide")
st.title("📊 DU COACHING RACE Analyzer")
st.info("This analyzer is brought to you by coach Davide Ambrosini")
st.markdown(
    """
    <div style="background-color:#7dd1d0; padding:10px; border-left:5px solid #a01fb4; border-radius:5px; color:#2c18db">
    This app is free but if you want to support my work you can 
    <a href="https://buymeacoffee.com/ultranerd" target="_blank" style="color:#a01fb4; font-weight:bold;">buy me a coffee</a>
    </div>
    """,
    unsafe_allow_html=True
)
# ----------------------------
# --- FIT FILE UPLOADER ---
# ----------------------------
st.write("")
st.markdown("*For large race files, to speed up the analysis, first add the race and cardiac data, and then upload the .fit file*")
uploaded_file = st.file_uploader("Upload a .fit file", type=["fit"])

if uploaded_file is not None:
    st.success("FIT file uploaded!")
    try:
        fitfile = FitFile(io.BytesIO(uploaded_file.getvalue()))
        records = []
        for rec in fitfile.get_messages("record"):
            row = {f.name: f.value for f in rec if f.name in ["timestamp","heart_rate","distance","enhanced_altitude","position_lat","position_long"]}
            if row:
                records.append(row)

        if not records:
            st.warning("⚠️ No usable records in this FIT file.")
            st.stop()

        df = pd.DataFrame(records)

        # Fill missing columns
        for col in ["heart_rate","distance","enhanced_altitude","position_lat","position_long"]:
            df[col] = df.get(col, np.nan)

        # Convert units
        df["distance_km"] = df["distance"].apply(lambda x: x/1000 if pd.notna(x) else np.nan)
        df["distance_km"] = df["distance_km"].ffill().fillna(0).fillna(0).astype(float)

        df["elevation_m"] = df["enhanced_altitude"].ffill().fillna(0).fillna(0).astype(float)

        df["lat"] = df["position_lat"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)
        df["lon"] = df["position_long"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)

        # --- Elapsed time safely ---
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_time = df["timestamp"].iloc[0]
            df["elapsed_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
        else:
            # fallback: crea sequenza numerica
            df["elapsed_sec"] = np.arange(len(df))

        # Ora possiamo calcolare in sicurezza time_diff_sec
        df["time_diff_sec"] = df["elapsed_sec"].diff().clip(lower=0).fillna(0)

        df["elapsed_hours"] = df["elapsed_sec"]/3600
        df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean() if "heart_rate" in df.columns else np.nan

        # Save df in session state so you can safely access later
        st.session_state['fit_df'] = df

        # Summary metrics
        kilometers = df["distance_km"].max() if "distance_km" in df.columns else 0
        total_elevation_gain = df["elevation_m"].diff().clip(lower=0).sum() if "elevation_m" in df.columns else 0
        st.session_state['kilometers'] = int(kilometers)
        st.session_state['total_elevation_gain'] = int(total_elevation_gain)

    except Exception as e:
        st.error(f"❌ Error reading FIT file: {e}")
else:
    st.info("👆 Please upload a .fit file to begin.")


# --- Athlete and race info form ---
with st.form("race_info_form"):
    athlete_name = st.text_input("🏃 Athlete's Name", value=st.session_state.get('athlete_name', ''))
    race_name = st.text_input("🏁 Race to be Analyzed", value=st.session_state.get('race_name', ''))
    race_date = st.date_input("📅 Date of the Race", value=st.session_state.get('race_date'))
    kilometers = st.session_state.get('kilometers', None)
    if kilometers is not None:
        st.markdown(f"📏 **Distance Run:** {kilometers} km")
    else:
        st.markdown("📏 **Distance Run:** _waiting for FIT file upload_")
    total_elevation_gain = st.session_state.get('total_elevation_gain', None)
    if total_elevation_gain is not None:
        st.markdown(f"🏔️ **Elevation gain:** {total_elevation_gain} m")
    else:
        st.markdown("🏔️ **Elevation gain:** _waiting for FIT file upload_")

    info_submitted = st.form_submit_button("Submit Info")

if info_submitted:
    st.session_state['athlete_name'] = athlete_name
    st.session_state['race_name'] = race_name
    st.session_state['race_date'] = race_date
    st.success("✅ Form submitted successfully!")

# --- Initialize default session_state values if missing ---
default_zones = {'z1': 140, 'z2': 160, 'z3': 170, 'z4': 180, 'z5': 200}
for zone, val in default_zones.items():
    if zone not in st.session_state:
        st.session_state[zone] = val

st.subheader("❤️ Athlete Heart Rate Zones")

# --- Input method ---
input_method = st.radio("Select input method:", ["Manual Input", "Import CSV"])

# --- Manual input ---
if input_method == "Manual Input":
    st.caption("Please input the *upper limit (in bpm)* for each training zone:")
    z1 = st.number_input("Zone 1 (Aerobic Low) - up to:", min_value=60, value=st.session_state['z1'])
    z2 = st.number_input("Zone 2 (Aerobic High) - up to:", min_value=60, value=st.session_state['z2'])
    z3 = st.number_input("Zone 3 (Aerobic Endurance) - up to:", min_value=60, value=st.session_state['z3'])
    z4 = st.number_input("Zone 4 (Sub Threshold) - up to:", min_value=60, value=st.session_state['z4'])
    z5 = st.number_input("Zone 5 (Super Threshold) - up to:", min_value=60, value=st.session_state['z5'])

# --- CSV import ---
elif input_method == "Import CSV":
    uploaded_hr_csv = st.file_uploader("Upload HR Zones CSV:", type=["csv"], key="hr_zones_csv")
    if uploaded_hr_csv is not None:
        hr_df = pd.read_csv(uploaded_hr_csv)
        required_cols = ['z1','z2','z3','z4','z5']
        if all(col in hr_df.columns for col in required_cols):
            z1, z2, z3, z4, z5 = hr_df.loc[0, required_cols]
            # Update session_state immediately
            st.session_state.update({col: hr_df.loc[0, col] for col in required_cols})
            athlete = hr_df.loc[0, 'athlete_name'] if 'athlete_name' in hr_df.columns else 'athlete'
            st.success(f"✅ Zones imported successfully for {athlete}")
        else:
            st.error("⚠️ CSV must contain columns: z1, z2, z3, z4, z5")
        # Sync local variables with session_state
        z1, z2, z3, z4, z5 = [st.session_state[col] for col in required_cols]

# --- Submit zones ---
if st.button("Submit HR Zones"):
    if not (z1 < z2 < z3 < z4 < z5):
        st.error("⚠️ There is something wrong in the HR data. Please correct the values.")
    else:
        # Save zones in session_state
        st.session_state.update({'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5})
        st.success("✅ Heart Rate Zones saved successfully!")

        st.write(f"""
        **HR Zones:**  
        - 🩵 Zone 1 (Aerobic Low): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic High): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Aerobic Endurance): {z2+1} - {z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1} - {z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1} - {z5} bpm
        """)

        # --- Export CSV ---
        athlete_name = st.session_state.get('athlete_name')
        if athlete_name:
            export_df = pd.DataFrame([{
                'athlete_name': athlete_name, 'z1': z1, 'z2': z2, 'z3': z3, 'z4': z4, 'z5': z5
            }])
            csv_data = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Export Zones to CSV for future analysis",
                data=csv_data,
                file_name=f"{athlete_name.replace(' ','_')}_HR_Zones.csv",
                mime='text/csv'
            )
        else:
            st.warning("⚠️ Please submit the Athlete Name in the race info form to export CSV.")

# --- SEGMENT SELECTION --- #
# --- Step 1: Select Number of Segments ---
with st.form("num_segments_form"):
    st.subheader("⏱️ Time segments for Time in Zone Analysis")
    st.caption("Step 1: Choose how many time segments you want to analyze")

    num_segments = st.number_input(
        "How many time segments do you want to analyze?",
        min_value=1, max_value=10, value=st.session_state.get('num_segments', 1), step=1
    )

    num_segments_submitted = st.form_submit_button("Save Number of Segments")

if num_segments_submitted:
    st.session_state['num_segments'] = num_segments
    st.success(f"✅ {num_segments} segment(s) selected!")

# --- Step 2: Input Segment Lengths (only shown after number is saved) ---
if st.session_state.get('num_segments'):
    with st.form("time_segment_form"):
        st.caption("Step 2: Define the start and end time (H:MM) for each segment")

        # --- Compute total duration safely ---
        total_duration_sec = None
        try:
            if 'fit_df' in st.session_state:
                _df = st.session_state['fit_df']
                if 'elapsed_sec' in _df.columns and not _df['elapsed_sec'].isna().all():
                    total_duration_sec = float(_df['elapsed_sec'].iloc[-1])
        except Exception:
            total_duration_sec = None

        def default_segment_time(i, n, total_sec, boundary="start"):
            """Return HH:MM string for segment i (1-indexed) boundary."""
            try:
                if total_sec is None or total_sec <= 0 or n <= 0:
                    raise ValueError
                segment_sec = total_sec / n
                if boundary == "start":
                    secs = segment_sec * (i - 1)
                else:
                    secs = segment_sec * i
                secs = int(round(secs))
                h = secs // 3600
                m = (secs % 3600) // 60
                return f"{h}:{m:02d}"
            except Exception:
                return "0:00" if boundary == "start" else "1:00"

        n = st.session_state['num_segments']
        segment_inputs = {}

        for i in range(1, n + 1):
            col1, col2 = st.columns(2)

            default_start = default_segment_time(i, n, total_duration_sec, "start")
            default_end   = default_segment_time(i, n, total_duration_sec, "end")

            with col1:
                segment_inputs[f'segment{i}_start'] = st.text_input(
                    f"Segment {i} Start",
                    value=st.session_state.get(f'segment{i}_start', default_start)
                )
            with col2:
                segment_inputs[f'segment{i}_end'] = st.text_input(
                    f"Segment {i} End",
                    value=st.session_state.get(f'segment{i}_end', default_end)
                )

        segments_submitted = st.form_submit_button("Save Time Segments")

    if segments_submitted:
        # Clear old segments first (in case user reduced the count)
        for i in range(1, 11):
            st.session_state.pop(f'segment{i}_start', None)
            st.session_state.pop(f'segment{i}_end', None)

        for key, val in segment_inputs.items():
            st.session_state[key] = val
        st.success("✅ Time segments saved successfully!")

# HELPER FOR LAP DETECTION
# --- Helper functions ---
def semicircles_to_degrees(s):
    return s * (180.0 / 2**31)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1-a)))

def seconds_to_hhmm(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h:02d}:{m:02d}"

def hhmm_to_seconds(t):
    try:
        h, m = t.split(":")
        return int(h) * 3600 + int(m) * 60
    except:
        return None

def nearest_idx(df, target_sec):
    return int((df["elapsed_sec"] - target_sec).abs().idxmin())

def sanitize_fit_df(df):
    """
    Clean FIT dataframe:
    - Ensure numeric columns are floats
    - Replace inf/-inf with NaN
    - Fill NaNs (forward-fill then zero)
    """
    df = df.copy()
    numeric_cols = ["elevation_m", "distance_km", "heart_rate"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")      # convert to float, NaN if bad
            df[col].replace([np.inf, -np.inf], np.nan, inplace=True)  # remove inf
            df[col] = df[col].ffill().fillna(0).fillna(0)     # fill NaNs

    # Ensure elapsed_sec exists
    if "elapsed_sec" not in df.columns:
        df["elapsed_sec"] = np.arange(len(df))
    return df

def detect_climbs(df, min_elev_gain=300, max_downhill=10):
    """
    Detect climbs automatically.
    Returns list of dicts: 'Climb Name', 'start_time', 'end_time'.
    """
    climbs = []
    in_climb = False
    climb_start_idx = None
    total_gain = 0.0
    temp_downhill = 0.0

    for i in range(1, len(df)):
        delta = float(df.loc[i, "elevation_m"]) - float(df.loc[i-1, "elevation_m"])

        if not in_climb:
            if delta > 0:
                in_climb = True
                climb_start_idx = i - 1
                total_gain = max(delta, 0.0)
                temp_downhill = 0.0
            continue

        if delta >= 0:
            total_gain += delta
            temp_downhill = 0.0
        else:
            temp_downhill += abs(delta)
            if temp_downhill > max_downhill:
                climb_end_idx = i - 1
                if total_gain >= min_elev_gain:
                    climbs.append({
                        "Climb Name": f"Climb {len(climbs)+1}",
                        "start_time": seconds_to_hhmm(int(df.loc[climb_start_idx, "elapsed_sec"])),
                        "end_time": seconds_to_hhmm(int(df.loc[climb_end_idx, "elapsed_sec"]))
                    })
                in_climb = False
                total_gain = 0.0
                temp_downhill = 0.0

    # Close open climb at end of data
    if in_climb and total_gain >= min_elev_gain:
        climbs.append({
            "Climb Name": f"Climb {len(climbs)+1}",
            "start_time": seconds_to_hhmm(int(df.loc[climb_start_idx, "elapsed_sec"])),
            "end_time": seconds_to_hhmm(int(df.loc[len(df)-1, "elapsed_sec"]))
        })

    return climbs

# ---------------------------
# Temporary metrics for preview table
# ---------------------------
def add_temporary_metrics(df, climbs):
    """
    Compute distance_km and elevation gain for temporary preview table.
    Read-only, no session_state modification.
    """
    temp = []
    for c in climbs:
        st_sec = hhmm_to_seconds(c["start_time"])
        end_sec = hhmm_to_seconds(c["end_time"])
        if st_sec is None or end_sec is None:
            c["distance_km"] = ""
            c["elev_gain_m"] = ""
            temp.append(c)
            continue

        start_idx = nearest_idx(df, st_sec)
        end_idx = nearest_idx(df, end_sec)
        if start_idx is None or end_idx is None or end_idx <= start_idx:
            c["distance_km"] = ""
            c["elev_gain_m"] = ""
            temp.append(c)
            continue

        d_dist = df.loc[end_idx, "distance_km"] - df.loc[start_idx, "distance_km"]
        seg = df["elevation_m"].iloc[start_idx:end_idx+1].diff()
        gain = seg[seg > 0].sum()
        c["distance_km"] = round(float(d_dist), 2)
        c["elev_gain_m"] = round(float(gain), 1)
        temp.append(c)
    return temp

# ---------------------------
# Compute processed climbs (after Save)
# ---------------------------
def compute_processed_climbs(df, edited_df):
    """
    Compute final distance, elevation, duration for edited climbs.
    Returns list for session_state["climb_data"].
    """
    processed = []
    for i, row in edited_df.iterrows():
        st_sec = hhmm_to_seconds(row["start_time"])
        end_sec = hhmm_to_seconds(row["end_time"])

        if st_sec is None or end_sec is None:
            raise ValueError(f"Row {i+1}: invalid time format. Use HH:MM.")

        start_idx = nearest_idx(df, st_sec)
        end_idx = nearest_idx(df, end_sec)

        if end_idx <= start_idx:
            raise ValueError(f"Row {i+1}: end time must be after start time.")

        d_dist = df.loc[end_idx, "distance_km"] - df.loc[start_idx, "distance_km"]
        seg = df["elevation_m"].iloc[start_idx:end_idx+1].diff()
        gain = seg[seg > 0].sum()

        processed.append({
            "name": row["Climb Name"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
            "duration": seconds_to_hhmm(end_sec - st_sec),
            "distance": round(float(d_dist), 2),
            "elevation": round(float(gain), 1),
            "ngp": ""
        })
    return processed

st.markdown("## 📋 Lap / Climb Analysis")
st.markdown("### Which analysis do you want to perform?")

# -----------------------------------------------------------
# 1️⃣ SELECT ANALYSIS TYPES (independent toggles)
# -----------------------------------------------------------

col_lap, col_climb = st.columns(2)

with col_lap:
    do_lap = st.checkbox(
        "🏃 Lap Analysis",
        value=st.session_state.get("do_lap_analysis", False),
        key="do_lap_checkbox"
    )

with col_climb:
    do_climb = st.checkbox(
        "⛰️ Climb Analysis",
        value=st.session_state.get("do_climb_analysis", False),
        key="do_climb_checkbox"
    )

st.session_state["do_lap_analysis"] = do_lap
st.session_state["do_climb_analysis"] = do_climb

# Keep a combined flag for downstream code that checks "do_lap_analysis"
# (lap OR climb active means some analysis is running)
st.session_state["any_analysis_active"] = do_lap or do_climb

# Reset climb insert method if climb is disabled
if not do_climb:
    st.session_state["climb_data_insert"] = None

# Reset lap insert method if lap is disabled
if not do_lap:
    st.session_state["lap_data_insert"] = None


# ===========================================================
# ⛰️ CLIMB ANALYSIS BLOCK
# ===========================================================

if do_climb:
    st.markdown("---")
    st.markdown("### ⛰️ Climb Analysis")

    climb_data_insert = st.radio(
        "How do you want to insert climb data?",
        ("Manually", "Automatic Climb Detector"),
        index=0,
        key="climb_data_insert_selector"
    )

    if climb_data_insert == "Manually":
        st.session_state["climb_data_insert"] = "manual"
    else:
        st.session_state["climb_data_insert"] = "automatic"

    # ---------------------------
    # AUTOMATIC CLIMB DETECTOR
    # ---------------------------
    if st.session_state.get("climb_data_insert") == "automatic":
        if "fit_df" not in st.session_state:
            st.warning("👆 Please upload a .fit file first.")
        else:
            df = st.session_state["fit_df"]

            min_elev_gain = st.number_input("Minimum elevation gain for a climb (meters)", value=300, key="climb_min_elev")
            max_downhill = st.number_input("Maximum allowed downhill inside a climb (meters)", value=5, key="climb_max_down")

            @st.cache_data
            def cached_detect_climbs(df, min_elev_gain, max_downhill):
                return detect_climbs(df, min_elev_gain=min_elev_gain, max_downhill=max_downhill)

            raw_climbs = cached_detect_climbs(df, min_elev_gain, max_downhill)
            if not raw_climbs:
                st.warning("No climbs detected with current parameters.")
            else:
                temp_climbs = add_temporary_metrics(df, [c.copy() for c in raw_climbs])

                if "editable_climb_table" not in st.session_state:
                    editable_df = pd.DataFrame(temp_climbs)
                    cols_desired = ["Climb Name", "start_time", "end_time", "distance_km", "elev_gain_m"]
                    for col in cols_desired:
                        if col not in editable_df.columns:
                            editable_df[col] = ""
                    editable_df = editable_df[cols_desired]
                    st.session_state["editable_climb_table"] = editable_df

                st.subheader("Automatically detected climbs — you can edit them below")
                x_preview = df["elapsed_sec"].apply(seconds_to_hhmm)
                fig_preview = go.Figure()
                fig_preview.add_trace(go.Scatter(
                    x=x_preview,
                    y=df["elevation_m"].astype(int),
                    mode="lines",
                    line=dict(color="gray"),
                    hovertemplate="Elapsed: %{x}<br>Elevation: %{y} m<extra></extra>"
                ))
                for c in temp_climbs:
                    st_sec = hhmm_to_seconds(c["start_time"])
                    end_sec = hhmm_to_seconds(c["end_time"])
                    if st_sec is None or end_sec is None:
                        continue
                    s = nearest_idx(df, st_sec)
                    e = nearest_idx(df, end_sec)
                    fig_preview.add_trace(go.Scatter(
                        x=x_preview[s:e+1],
                        y=df["elevation_m"].iloc[s:e+1].astype(int),
                        mode="lines",
                        line=dict(color="green"),
                        fill="tozeroy",
                        opacity=0.4,
                        hovertemplate="Elapsed: %{x}<br>Elevation: %{y} m<extra></extra>"
                    ))
                st.plotly_chart(fig_preview, use_container_width=True)

                @st.fragment
                def climb_table_fragment():
                    st.subheader("Detected Climbs")
                    st.info("You can edit climb names, start times, and end times")
                    edited = st.data_editor(
                        st.session_state["editable_climb_table"],
                        num_rows="dynamic",
                        key="climb_editor",
                        disabled=["distance_km", "elev_gain_m"]
                    )
                    st.session_state["editable_climb_table"] = edited

                climb_table_fragment()

                if st.button("Save climbs and compute metrics", key="save_climbs_auto"):
                    try:
                        processed_climbs = compute_processed_climbs(df, st.session_state["editable_climb_table"])
                        st.session_state["climb_data"] = processed_climbs
                        st.success("✅ Climbs saved and metrics computed!")
                    except ValueError as e:
                        st.error(str(e))

                if "climb_data" in st.session_state:
                    final_climbs = st.session_state["climb_data"]
                    x_axis_option_final = st.radio(
                        "X-axis for climb plot:",
                        ["Elapsed Time", "Distance (km)"],
                        key="xaxis_final"
                    )
                    if x_axis_option_final == "Elapsed Time":
                        x_final = df["elapsed_sec"].apply(seconds_to_hhmm)
                        x_label = "Elapsed Time (HH:MM)"
                    else:
                        x_final = df["distance_km"]
                        x_label = "Distance (km)"

                    fig_final = go.Figure()
                    fig_final.add_trace(go.Scatter(
                        x=x_final,
                        y=df["elevation_m"].astype(int),
                        mode="lines",
                        line=dict(color="gray"),
                        hovertemplate=f"{x_label}: %{{x}}<br>Elevation: %{{y}} m<extra></extra>"
                    ))
                    for c in final_climbs:
                        s = c["start_idx"]
                        e = c["end_idx"]
                        fig_final.add_trace(go.Scatter(
                            x=x_final[s:e+1],
                            y=df["elevation_m"].iloc[s:e+1].astype(int),
                            mode="lines",
                            line=dict(color="green"),
                            fill="tozeroy",
                            opacity=0.4,
                            hovertemplate=f"{x_label}: %{{x}}<br>Elevation: %{{y}} m<extra></extra>"
                        ))
                    fig_final.update_layout(
                        title="Selected climbs highlighted",
                        hovermode="x unified",
                        showlegend=False
                    )
                    fig_final.update_yaxes(title_text="Elevation (m)", tickformat="d")
                    st.plotly_chart(fig_final, use_container_width=True)

                    final_df = pd.DataFrame(final_climbs)
                    display_cols = ["name", "start_time", "end_time", "duration", "distance", "elevation"]
                    st.dataframe(final_df[[c for c in display_cols if c in final_df.columns]].reset_index(drop=True))

    # ---------------------------
    # MANUAL CLIMB INPUT
    # ---------------------------
    elif st.session_state.get("climb_data_insert") == "manual":
        if "fit_df" not in st.session_state:
            st.warning("👆 Please upload a .fit file first.")
        else:
            df = st.session_state["fit_df"]

            st.subheader("Climb Elevation Profile")
            fig_manual = go.Figure()
            fig_manual.add_trace(go.Scatter(
                x=df["elapsed_sec"].apply(seconds_to_hhmm),
                y=df["elevation_m"].astype(int),
                mode="lines",
                line=dict(color="gray"),
                hovertemplate="Elapsed: %{x}<br>Elevation: %{y} m<extra></extra>"
            ))
            st.plotly_chart(fig_manual, use_container_width=True)

            table_key = "manual_climb_table"
            if table_key not in st.session_state:
                st.session_state[table_key] = pd.DataFrame(columns=["name", "start_time", "end_time", "ngp"])

            st.subheader("Add/Edit Climbs")
            st.info("Climbs have to be added in the table below with HH:MM format")

            with st.form("manual_climb_form"):
                edited = st.data_editor(
                    st.session_state[table_key],
                    num_rows="dynamic",
                    key="manual_climb_editor"
                )
                submit = st.form_submit_button("Save manual Climbs")

                if submit:
                    edited = edited.reset_index(drop=True)

                    # --- Validation ---
                    incomplete_rows = []
                    for i, row in edited.iterrows():
                        if pd.isna(row.get("start_time")) or str(row.get("start_time", "")).strip() == "":
                            incomplete_rows.append(f"Row {i+1}: missing start time")
                        if pd.isna(row.get("end_time")) or str(row.get("end_time", "")).strip() == "":
                            incomplete_rows.append(f"Row {i+1}: missing end time")

                    if incomplete_rows:
                        for msg in incomplete_rows:
                            st.error(f"⚠️ {msg}")
                        st.stop()

                    # --- Only save if all rows are valid ---
                    edited["name"] = [
                        row["name"] if pd.notna(row.get("name")) and str(row["name"]).strip() != "" else f"Climb {i+1}"
                        for i, row in edited.iterrows()
                    ]

                    # Only update session state if there's actual data
                    if not edited.empty:
                        st.session_state[table_key] = edited
                    
                    climb_data = []
                    for _, row in edited.iterrows():
                        if not row["start_time"] or not row["end_time"]:
                            continue
                        climb_data.append({
                            "name": row["name"],
                            "start_time": row["start_time"],
                            "end_time": row["end_time"],
                            "ngp": row.get("ngp", "")
                        })

                    if climb_data:
                        st.session_state["climb_data"] = climb_data
                        st.success(f"✅ {len(climb_data)} climb(s) saved successfully!")
                    else:
                        st.error("⚠️ No valid climbs to save. Please fill in start and end times.")

            if st.session_state.get("climb_data"):
                fig_final = go.Figure()
                fig_final.add_trace(go.Scatter(
                    x=df["elapsed_sec"].apply(seconds_to_hhmm),
                    y=df["elevation_m"].astype(int),
                    mode="lines",
                    line=dict(color="gray")
                ))
                for entry in st.session_state["climb_data"]:
                    st_sec = hhmm_to_seconds(entry["start_time"])
                    end_sec = hhmm_to_seconds(entry["end_time"])
                    if st_sec is None or end_sec is None:
                        continue
                    mask = (df["elapsed_sec"] >= st_sec) & (df["elapsed_sec"] <= end_sec)
                    df_segment = df.loc[mask]
                    if df_segment.empty:
                        continue
                    fig_final.add_trace(go.Scatter(
                        x=df_segment["elapsed_sec"].apply(seconds_to_hhmm),
                        y=df_segment["elevation_m"].astype(int),
                        mode="lines",
                        line=dict(color="green"),
                        fill="tozeroy",
                        opacity=0.4
                    ))
                fig_final.update_layout(
                    title="Manual Climbs Highlighted",
                    hovermode="x unified",
                    showlegend=False
                )
                st.plotly_chart(fig_final, use_container_width=True)
                st.dataframe(pd.DataFrame(st.session_state["climb_data"]).reset_index(drop=True))


# ===========================================================
# 🏃 LAP ANALYSIS BLOCK
# ===========================================================

if do_lap:
    st.markdown("---")
    st.markdown("### 🏃 Lap Analysis")

    lap_data_insert = st.radio(
        "How do you want to insert lap data?",
        ("Manually", "Automatic Lap detector", "Distance slicer"),
        index=0,
        key="lap_data_insert_selector"
    )

    if lap_data_insert == "Manually":
        st.session_state["lap_data_insert"] = "manual"
    elif lap_data_insert == "Automatic Lap detector":
        st.session_state["lap_data_insert"] = "automatic"
    else:
        st.session_state["lap_data_insert"] = "fix_distance"

    # ---------------------------
    # AUTOMATIC LOOP DETECTOR
    # ---------------------------
    loops = []

    if st.session_state.get("lap_data_insert") == "automatic":
        if uploaded_file is None or "fit_df" not in st.session_state:
            st.warning("👆 Please upload a .fit file first.")
        else:
            df = st.session_state["fit_df"]

            col1, col2, col3 = st.columns(3)
            with col1:
                base_radius = st.number_input("Base radius (meters)", value=20, min_value=1)
            with col2:
                percent_error = st.slider("GPS antenna error (%)", 0.0, 5.0, 1.0) / 100
            with col3:
                min_samples_between_crossings = st.slider("Min samples between crossings", 1, 50, 5)

            required_cols = ["lat", "lon", "timestamp", "elevation_m", "distance_km", "elapsed_sec"]
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns: {', '.join([c for c in required_cols if c not in df.columns])}")
            else:
                df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)
                df["elevation_m"] = df["elevation_m"].astype(float)
                df["distance_km"] = df["distance_km"].astype(float)

                from haversine import haversine

                start_lat, start_lon = df.loc[0, ["lat", "lon"]]
                df["dist_from_start_m"] = df.apply(
                    lambda r: haversine((start_lat, start_lon), (r["lat"], r["lon"])) * 1000,
                    axis=1
                )
                threshold = base_radius + percent_error * df["dist_from_start_m"].max()

                crossings = []
                inside = df.loc[0, "dist_from_start_m"] <= threshold
                last_crossing_idx = 0 if inside else -1
                if inside:
                    crossings.append((0, df.loc[0, "timestamp"]))

                for idx in range(1, len(df)):
                    d = df.loc[idx, "dist_from_start_m"]
                    currently_inside = d <= threshold
                    if (not inside) and currently_inside:
                        if (idx - last_crossing_idx) > min_samples_between_crossings:
                            crossings.append((idx, df.loc[idx, "timestamp"]))
                            last_crossing_idx = idx
                    inside = currently_inside

                if len(crossings) > 1:
                    for i in range(1, len(crossings)):
                        prev_idx, _ = crossings[i - 1]
                        idx, _ = crossings[i]
                        df_lap = df.iloc[prev_idx:idx + 1].copy()
                        start_sec = df_lap["elapsed_sec"].iloc[0]
                        end_sec = df_lap["elapsed_sec"].iloc[-1]
                        duration_sec = end_sec - start_sec
                        loops.append({
                            "name": f"Lap {i}",
                            "start_time": seconds_to_hhmm(start_sec),
                            "end_time": seconds_to_hhmm(end_sec),
                            "duration": seconds_to_hhmm(duration_sec),
                            "start_idx": prev_idx,
                            "end_idx": idx,
                            "distance": round(df_lap["distance_km"].max() - df_lap["distance_km"].min(), 1),
                            "elevation": int(df_lap["elevation_m"].diff().clip(lower=0).sum()),
                            "ngp": ""
                        })

                st.session_state["lap_data"] = loops
                st.session_state["lap_form_submitted"] = True

                st.subheader("Detected Loops")
                if loops:
                    loops_df = pd.DataFrame(loops)
                    loops_df["distance"] = loops_df["distance"].astype(float).map("{:.1f}".format)
                    st.dataframe(loops_df[["name", "start_time", "end_time", "duration", "distance", "elevation"]])
                else:
                    st.warning("No loops detected.")

    # ---------------------------
    # DISTANCE SLICER
    # ---------------------------
    elif st.session_state.get("lap_data_insert") == "fix_distance":
        if uploaded_file is None or "fit_df" not in st.session_state:
            st.info("👆 Please upload a .fit file first to use the Distance Slicer.")
        else:
            df = st.session_state["fit_df"]
            max_distance = df["distance_km"].max()
            slice_distance = st.number_input(
                "How many km would you like each lap to be?",
                min_value=1,
                max_value=int(math.ceil(max_distance)),
                value=10
            )

            if st.button("Generate Laps", key="generate_distance_laps"):
                laps = []
                start_dist = 0.0
                lap_num = 1
                total_distance = round(max_distance, 3)

                while start_dist < total_distance:
                    end_dist = min(start_dist + slice_distance, total_distance)
                    lap_df = df[
                        (df["distance_km"].round(3) >= round(start_dist, 3)) &
                        (df["distance_km"].round(3) <= round(end_dist, 3))
                    ].copy()
                    if lap_df.empty:
                        break

                    start_sec = lap_df["elapsed_sec"].iloc[0]
                    end_sec = lap_df["elapsed_sec"].iloc[-1]
                    duration_sec = end_sec - start_sec

                    laps.append({
                        "name": f"Lap {lap_num}",
                        "start_time": seconds_to_hhmm(start_sec),
                        "end_time": seconds_to_hhmm(end_sec),
                        "duration": seconds_to_hhmm(duration_sec),
                        "start_idx": lap_df.index[0],
                        "end_idx": lap_df.index[-1],
                        "distance": round(end_dist - start_dist, 3)
                    })
                    start_dist += slice_distance
                    lap_num += 1

                st.session_state["lap_data"] = laps
                st.session_state["lap_form_submitted"] = True

                st.subheader("Distance Slicer Laps")
                if laps:
                    st.dataframe(pd.DataFrame(laps)[["name", "start_time", "end_time", "duration", "distance"]])
                else:
                    st.warning("No laps generated. Check the distance slice or your FIT file.")

# ---------------------------
# MANUAL LAP INPUT (distance-based)
# ---------------------------

    elif st.session_state.get("lap_data_insert") == "manual":
        if "fit_df" not in st.session_state:
            st.warning("👆 Please upload a .fit file first.")
        else:
            df = st.session_state["fit_df"]
            max_distance = round(df["distance_km"].max(), 2)

            # --- Helper: find nearest row index for a given distance ---
            def nearest_idx_by_distance(df, target_km):
                return int((df["distance_km"] - target_km).abs().idxmin())

            # --- Elevation profile (x = distance) ---
            st.subheader("Lap Elevation Profile")
            fig_manual = go.Figure()
            fig_manual.add_trace(go.Scatter(
                x=df["distance_km"],
                y=df["elevation_m"].astype(int),
                mode="lines",
                line=dict(color="gray"),
                hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y} m<extra></extra>"
            ))
            fig_manual.update_layout(
                xaxis_title="Distance (km)",
                yaxis_title="Elevation (m)"
            )
            st.plotly_chart(fig_manual, use_container_width=True)

            st.caption(f"Total race distance: **{max_distance} km**")

            # --- Editable table: user inputs distance boundaries ---
            table_key = "manual_lap_table"
            if table_key not in st.session_state:
                st.session_state[table_key] = pd.DataFrame(
                    columns=["name", "start_km", "end_km", "ngp"]
                )

            st.subheader("Add/Edit Laps")
            st.info(f"Enter the start and end distance in km for each lap (max: {max_distance} km)")

            with st.form("manual_lap_form"):
                edited = st.data_editor(
                    st.session_state[table_key],
                    num_rows="dynamic",
                    key="manual_lap_editor",
                    column_config={
                        "name": st.column_config.TextColumn("Lap Name"),
                        "start_km": st.column_config.NumberColumn(
                            "Start (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                        ),
                        "end_km": st.column_config.NumberColumn(
                            "End (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                        ),
                        "ngp": st.column_config.TextColumn("NGP"),
                    }
                )
                submit = st.form_submit_button("Save manual Laps")

                if submit:
                    edited = edited.reset_index(drop=True)

                    # --- Validation ---
                    incomplete_rows = []
                    for i, row in edited.iterrows():
                        if pd.isna(row.get("start_km")):
                            incomplete_rows.append(f"Row {i+1}: missing start km")
                        if pd.isna(row.get("end_km")):
                            incomplete_rows.append(f"Row {i+1}: missing end km")
                        elif not pd.isna(row.get("start_km")) and float(row["end_km"]) <= float(row["start_km"]):
                            incomplete_rows.append(f"Row {i+1}: end km must be greater than start km")

                    if incomplete_rows:
                        for msg in incomplete_rows:
                            st.error(f"⚠️ {msg}")
                        st.stop()

                    # --- Only save if there's actual data ---
                    edited["name"] = [
                        row["name"] if pd.notna(row.get("name")) and str(row["name"]).strip() != "" else f"Lap {i+1}"
                        for i, row in edited.iterrows()
                    ]

                    if not edited.empty:
                        st.session_state[table_key] = edited

                    lap_data = []
                    errors = []
                    for i, row in edited.iterrows():
                        start_km = float(row["start_km"])
                        end_km   = float(row["end_km"])

                        if end_km > max_distance:
                            errors.append(f"Row {i+1} ({row['name']}): end km ({end_km}) exceeds race distance ({max_distance} km).")
                            continue

                        start_idx    = nearest_idx_by_distance(df, start_km)
                        end_idx      = nearest_idx_by_distance(df, end_km)
                        start_sec    = df.loc[start_idx, "elapsed_sec"]
                        end_sec      = df.loc[end_idx,   "elapsed_sec"]
                        duration_sec = end_sec - start_sec
                        lap_slice    = df.loc[start_idx:end_idx]
                        elevation    = int(lap_slice["elevation_m"].diff().clip(lower=0).sum())

                        lap_data.append({
                            "name":       row["name"],
                            "start_km":   round(start_km, 2),
                            "end_km":     round(end_km, 2),
                            "distance":   round(end_km - start_km, 2),
                            "start_time": seconds_to_hhmm(start_sec),
                            "end_time":   seconds_to_hhmm(end_sec),
                            "duration":   seconds_to_hhmm(duration_sec),
                            "elevation":  elevation,
                            "start_idx":  start_idx,
                            "end_idx":    end_idx,
                            "ngp":        row.get("ngp", "")
                        })

                    for err in errors:
                        st.error(err)

                    if lap_data:
                        st.session_state["lap_data"] = lap_data
                        st.session_state["lap_form_submitted"] = True
                        st.success(f"✅ {len(lap_data)} lap(s) saved!")
                    else:
                        st.error("⚠️ No valid laps to save. Please check your distance values.")

            # --- Preview: highlight laps on elevation profile ---
            if st.session_state.get("lap_form_submitted") and st.session_state.get("lap_data"):
                fig_final = go.Figure()
                fig_final.add_trace(go.Scatter(
                    x=df["distance_km"],
                    y=df["elevation_m"].astype(int),
                    mode="lines",
                    line=dict(color="gray"),
                    hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y} m<extra></extra>"
                ))

                for entry in st.session_state["lap_data"]:
                    s = entry["start_idx"]
                    e = entry["end_idx"]
                    df_segment = df.loc[s:e]
                    if df_segment.empty:
                        continue
                    fig_final.add_trace(go.Scatter(
                        x=df_segment["distance_km"],
                        y=df_segment["elevation_m"].astype(int),
                        mode="lines",
                        line=dict(color="green"),
                        fill="tozeroy",
                        opacity=0.4,
                        name=entry["name"],
                        hovertemplate=f"{entry['name']}<br>Distance: %{{x:.2f}} km<br>Elevation: %{{y}} m<extra></extra>"
                    ))

                fig_final.update_layout(
                    title="Manual Laps Highlighted",
                    xaxis_title="Distance (km)",
                    yaxis_title="Elevation (m)",
                    hovermode="x unified",
                    showlegend=True
                )
                st.plotly_chart(fig_final, use_container_width=True)

                # Summary table
                display_cols = ["name", "start_km", "end_km", "distance", "start_time", "end_time", "duration", "elevation"]
                st.dataframe(
                    pd.DataFrame(st.session_state["lap_data"])[[c for c in display_cols]].reset_index(drop=True)
                )
#-------------
# ------- ANALYSIS START
#-------------

# ---- page slicer
st.markdown(
    """
    <hr style="border:1px solid #336699">
    <h4 style='text-align:center; color:white;'>📊 Race Analysis</h4>
    <hr style="border:1px solid #336699">
    """,
    unsafe_allow_html=True
)

# ----- analysis start -----#

if uploaded_file is None:
    st.info("👆 Upload a FIT file to run the analysis")
else:
    # --- Athlete & race info display ---
    if 'athlete_name' not in st.session_state:
        st.warning("⚠️ Please submit the Athlete and Race info in the form above")
    else:
        st.markdown("---")
        st.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
        st.markdown(f"**Race:** {st.session_state['race_name']}")
        formatted_date = st.session_state['race_date'].strftime("%d/%m/%Y")
        st.markdown(f"**Date:** {formatted_date}")
        st.markdown(f"📏 **Distance Run:** {kilometers} km")
        st.markdown(f"🏔️ **Elevation gain:** {total_elevation_gain} m")

        total_seconds = df["elapsed_sec"].iloc[-1]
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        final_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
        st.markdown(f"**Final Time:** {final_time_str}")
        st.markdown("---")
        st.write(f"""
        **HR Zones:**  
        - 🩵 Zone 1 (Aerobic Low): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic High): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Aerobic Endurance): {z2+1} - {z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1} - {z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1} - {z5} bpm
        """)
        st.markdown("---")

    # --- Smooth HR ---
    df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean()

    # --- HR averages ---
    mid_index = len(df) // 2
    first_half_avg = df["heart_rate"][:mid_index].mean()
    second_half_avg = df["heart_rate"][mid_index:].mean()
    overall_avg = df["heart_rate"].mean()
    percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

    st.markdown(f"❤️ Overall Average HR: **{overall_avg:.0f} bpm**")
    st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
    st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")
    if percent_diff >= -10:
        st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
    else:
        st.error(f"📊 % Difference: **{percent_diff:.1f}%**")


#--------------------#
# --- DET Index ---
#------------------------#

if uploaded_file is not None:

    # --- Clean data ---
    df_clean = df.dropna(subset=["hr_smooth", "elapsed_sec"])
    df_clean["elapsed_hours"] = df_clean["elapsed_sec"] / 3600

    X = df_clean["elapsed_sec"].values.reshape(-1,1)
    y = df_clean["hr_smooth"].values

    # --- Linear regression for trend line ---
    reg = LinearRegression().fit(X, y)
    df_clean["trend_line"] = reg.predict(X)

    # --- DET index calculation ---
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
    tooltip_text = ("DI < 4 - SCARSO DECADIMENTO\nDI = 7 - DECADIMENTO MEDIO\nDI > 10 - ALTO DECADIMENTO")
    st.markdown(
        f"<div title='{tooltip_text}' style='font-size:16px; background-color:{color}; color:black; padding:5px; border-radius:5px; display:inline-block;'>📈 DET INDEX: <b>{det_index_str}</b> ({comment})</div>", 
        unsafe_allow_html=True
    )
# -------------------------#
# ---- LIVE CHARTS ------------- #

if uploaded_file is not None:

    # --- Prepare hover info ---
    df_clean["Race Time [h:mm]"] = df_clean.apply(
        lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm",
        axis=1
    )

    # --- Plotly chart ---
    fig = px.line(
        df_clean,
        x="elapsed_hours",
        y="hr_smooth",
        labels={"elapsed_hours":"Elapsed Time (hours)", "hr_smooth":"Heart Rate (bpm)"},
        title="Heart Rate Over Time",
        hover_data={"Race Time [h:mm]": True, "elapsed_hours": False, "hr_smooth": False}
    )

    # Add trend line
    fig.add_scatter(
        x=df_clean["elapsed_hours"],
        y=df_clean["trend_line"],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Trend Line'
    )

    # Hover template for HR line
    fig.update_traces(hovertemplate='%{customdata[0]}', selector=dict(name='hr_smooth'))

    # Y-axis format
    fig.update_yaxes(tickformat='d')

    st.plotly_chart(fig, use_container_width=True)


# ------------------------------------------
# SEGMENT ANALYSIS #
# ------------------------------------------

    st.markdown("## SEGMENT ANALYSIS")

    # Time in zone analysis for segments

    # Check conditions

    def format_hmm(hmm):
        """Convert H:MM to display format: 0 for 0:00, or 1h00' for 1:00 etc."""
        try:
            h, m = hmm.split(":")
            h, m = int(h), int(m)
            if h == 0 and m == 0:
                return "0"
            else:
                return f"{h}h{m:02d}'"
        except:
            return hmm

    if all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):
        z1, z2, z3, z4, z5 = st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5']

        def get_hr_zone(hr):
            if hr <= z1:
                return "Zone 1 // Aerobic Low"
            elif hr <= z2:
                return "Zone 2 // Aerobic High"
            elif hr <= z3:
                return "Zone 3 // Aerobic Endurance"
            elif hr <= z4:
                return "Zone 4 // Sub Threshold"
            else:
                return "Zone 5 // Super Threshold"
        
        df["HR Zone"] = df["heart_rate"].apply(get_hr_zone)
        df["time_diff_sec"] = df["elapsed_sec"].diff().clip(lower=0).fillna(0)

        zone_order = ["Zone 1 // Aerobic Low","Zone 2 // Aerobic High","Zone 3 // Aerobic Endurance","Zone 4 // Sub Threshold","Zone 5 // Super Threshold"]

        # Total (overall) time-in-zone
        total_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)

        # Time segment warning

        _n = st.session_state.get('num_segments', 1)
        segment_keys = [f'segment{i}_{se}' for i in range(1, _n+1) for se in ['start','end']]

        if not all(k in st.session_state for k in segment_keys):
            missing_seg = [k for k in segment_keys if k not in st.session_state]
            st.warning(f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis.")

        # Prepare segment inputs with formatted names
        segment_inputs = []    
        for i in range(1, st.session_state.get('num_segments', 1) + 1):
            start_key = f'segment{i}_start'
            end_key = f'segment{i}_end'
            if all(k in st.session_state for k in [start_key, end_key]):
                seg_name = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                segment_inputs.append(
                    (st.session_state[start_key], st.session_state[end_key], seg_name)
                )
        
        segment_data = {}

        def h_mm_to_seconds(hmm):
            try:
                h, m = hmm.split(":")
                return int(h)*3600 + int(m)*60
            except:
                return None

        for start_str, end_str, seg_name in segment_inputs:
            start_sec = h_mm_to_seconds(start_str)
            end_sec = h_mm_to_seconds(end_str)
            if start_sec is None or end_sec is None or start_sec >= end_sec:
                segment_data[seg_name] = pd.Series(0, index=zone_order)
                continue

            df_segment = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()
            df_segment["time_diff_sec"] = df_segment["elapsed_sec"].diff().fillna(0)
            seg_summary = df_segment.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
            segment_data[seg_name] = seg_summary

        # Combine all segments + total into one DataFrame
        combined_df = pd.DataFrame(segment_data)
        combined_df["Total"] = total_summary

        # Format only numeric columns as H:MM
        num_cols = combined_df.select_dtypes(include=["number"]).columns
        for col in num_cols:
            combined_df[col] = combined_df[col].apply(
                lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}" if pd.notna(x) else ""
            )

        st.markdown("### ⏱️ Time-in-Zone Analysis")
        st.dataframe(combined_df)

    else:
        st.warning("⚠️ Please submit the Heart Rate Zones to enable Time-in-Zone analysis.")    


# --- Elevation Profile with Time Segments (Live Chart) ---
if uploaded_file is not None and 'df' in locals() and not df.empty:

    segment_colors = ["royalblue", "tomato", "gold", "mediumseagreen", "orchid",
                      "darkorange", "deepskyblue", "limegreen", "crimson", "slateblue"]

    x_hhmm = df["elapsed_sec"].apply(seconds_to_hhmm)
    elevation_smooth = df["elevation_m"].rolling(window=20, min_periods=1).mean()

    fig_seg = go.Figure()

    # Segment filled areas — added FIRST so elevation line renders on top
    for i in range(1, st.session_state.get('num_segments', 1) + 1):
        start_key = f'segment{i}_start'
        end_key   = f'segment{i}_end'
        if all(k in st.session_state for k in [start_key, end_key]):
            seg_start = h_mm_to_seconds(st.session_state[start_key])
            seg_end   = h_mm_to_seconds(st.session_state[end_key])
            if seg_start is not None and seg_end is not None and seg_end > seg_start:
                mask = (df["elapsed_sec"] >= seg_start) & (df["elapsed_sec"] <= seg_end)
                seg_label = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                color = segment_colors[(i - 1) % len(segment_colors)]
                fig_seg.add_trace(go.Scatter(
                    x=x_hhmm[mask],
                    y=elevation_smooth[mask],
                    mode="none",
                    fill="tozeroy",
                    fillcolor=color,
                    opacity=0.3,
                    name=seg_label,
                    hoverinfo="skip"
                ))

    # Base elevation line — always on top
    fig_seg.add_trace(go.Scatter(
        x=x_hhmm,
        y=elevation_smooth,
        mode="lines",
        line=dict(color="lightskyblue", width=1.5),
        name="Elevation",
        hovertemplate="Time: %{x}<br>Elevation: %{y:.0f} m<extra></extra>"
    ))

    fig_seg.update_layout(
        title="Elevation Profile with Time Segments Highlighted",
        xaxis_title="Elapsed Time (hh:mm)",
        yaxis_title="Elevation (m)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=80)
    )
    fig_seg.update_yaxes(tickformat="d")
    fig_seg.update_xaxes(tickangle=45)

    st.plotly_chart(fig_seg, use_container_width=True)
    
    # GROUPED BAR CHART FOR TIME IN ZONE
if uploaded_file is not None and 'HR Zone' in df.columns:
    missing_seg = [k for k in segment_keys if k not in st.session_state]
    if missing_seg:
        st.warning(
            f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis"
        )

    bar_df = combined_df.copy()
    bar_df.index = [f"Zone {i+1}" for i in range(len(bar_df))]
    bar_df_reset = bar_df.reset_index().rename(columns={'index':'HR Zone'})
    bar_long = bar_df_reset.melt(id_vars="HR Zone", var_name="Segment", value_name="Time [h:mm]")

    # Convert H:MM to hours float
    def h_mm_to_float(hmm_str):
        try:
            h, m = map(int, hmm_str.split(":"))
            return h + m/60
        except:
            return 0

    bar_long["Hours"] = bar_long["Time [h:mm]"].apply(h_mm_to_float)

    fig_bar = px.bar(
        bar_long,
        x="HR Zone",
        y="Hours",
        color="Segment",
        barmode="group",
        hover_data={"Segment": True, "Time [h:mm]": True, "Hours": False, "HR Zone": True},
        title="⏱️ Time-in-Zone per Segment (Bar Chart)"
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)

# =====================================
# 📊 HR DENSITY DISTRIBUTION BY ZONE
# =====================================

if uploaded_file is not None and 'HR Zone' in df.columns and all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):

    st.markdown("### 📊 Heart Rate Density Distribution by Zone")

    # --- Zone boundaries ---
    _zone_bands = [
        {"name": "Z1", "x0": 0,   "x1": z1, "color": "rgba(100, 200, 255, 0.15)"},
        {"name": "Z2", "x0": z1, "x1": z2, "color": "rgba(100, 220, 100, 0.15)"},
        {"name": "Z3", "x0": z2, "x1": z3, "color": "rgba(255, 230, 50,  0.15)"},
        {"name": "Z4", "x0": z3, "x1": z4, "color": "rgba(255, 150, 50,  0.15)"},
        {"name": "Z5", "x0": z4, "x1": z5, "color": "rgba(255, 80,  80,  0.15)"},
    ]

    def build_density_chart(hr_data, title, avg_hr, z1, z2, z3, z4, z5, zone_bands, x_min=None, x_max=None):
        kde = gaussian_kde(hr_data, bw_method=0.3)
        x_range = np.linspace(hr_data.min(), hr_data.max(), 500)
        y_kde = kde(x_range)
        y_kde = (y_kde / y_kde.sum()) * 100

        # Use provided range or fall back to data-derived range
        if x_min is None:
            x_min = np.percentile(hr_data, 1)
        if x_max is None:
            x_max = hr_data.max()

        fig = go.Figure()

        # Zone background bands
        for zone in zone_bands:
            fig.add_vrect(
                x0=zone["x0"],
                x1=zone["x1"],
                fillcolor=zone["color"],
                layer="below",
                line_width=0,
            )
            band_center = (zone["x0"] + zone["x1"]) / 2
            if zone["name"] == "Z1":
                label_x = z1 - 10
            else:
                label_x = max(x_min, min(band_center, x_max))
            fig.add_annotation(
                x=label_x,
                y=1.02,
                xref="x",
                yref="paper",
                text=zone["name"],
                showarrow=False,
                font=dict(size=15, color="black", weight="bold"),
                xanchor="center"
            )

        # KDE smooth curve
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_kde,
            mode="lines",
            fill="tozeroy",
            line=dict(color="rgba(50, 120, 200, 1)", width=2),
            fillcolor="rgba(50, 120, 200, 0.3)",
            name="HR Density",
            hovertemplate="HR: %{x:.0f} bpm<br>Probability: %{y:.1f}%<extra></extra>"
        ))

        # Vertical zone boundary lines
        for bpm in [z1, z2, z3, z4, z5]:
            fig.add_vline(
                x=bpm,
                line_dash="dash",
                line_color="gray",
                line_width=1
            )

        # AVG RACE BPM line
        fig.add_vline(
            x=avg_hr,
            line_dash="solid",
            line_color="rgba(200, 50, 50, 1)",
            line_width=3
        )
        fig.add_annotation(
            x=avg_hr,
            y=0.97,
            xref="x",
            yref="paper",
            text=f"AVG RACE BPM<br><b>{avg_hr:.0f} bpm</b>",
            showarrow=False,
            font=dict(size=12, color="rgba(200, 50, 50, 1)"),
            bgcolor="white",
            bordercolor="rgba(200, 50, 50, 0.4)",
            borderwidth=1,
            xanchor="left",
            yanchor="top"
        )

        fig.update_layout(
            title=title,
            xaxis_title="Heart Rate (bpm)",
            yaxis_title="Density %",
            showlegend=False,
            plot_bgcolor="white",
            margin=dict(t=80),
            xaxis=dict(range=[x_min, x_max])
        )

        return fig

    # --- Compute full race avg ONCE ---
    _hr_data_total = df["heart_rate"].dropna()
    _avg_hr_total = _hr_data_total.mean()
    _x_min_total = np.percentile(_hr_data_total, 1)
    _x_max_total = _hr_data_total.max()

    # --- Full race chart ---
    if len(_hr_data_total) > 1:
        st.plotly_chart(
            build_density_chart(
                _hr_data_total,
                "Heart Rate Density Distribution - Full Race",
                _avg_hr_total,
                z1, z2, z3, z4, z5,
                _zone_bands
            ),
            use_container_width=True
        )

    # --- Per-segment charts ---
    _segment_inputs = []
    for _i in range(1,  st.session_state.get('num_segments', 1) + 1):
        _start_key = f'segment{_i}_start'
        _end_key = f'segment{_i}_end'
        if all(k in st.session_state for k in [_start_key, _end_key]):
            _start_str = st.session_state[_start_key]
            _end_str = st.session_state[_end_key]
            _start_sec = h_mm_to_seconds(_start_str)
            _end_sec = h_mm_to_seconds(_end_str)
            if _start_sec is not None and _end_sec is not None and _end_sec > _start_sec:
                _seg_name = f"{format_hmm(_start_str)} to {format_hmm(_end_str)}"
                _segment_inputs.append((_start_sec, _end_sec, _seg_name))

    for _start_sec, _end_sec, _seg_name in _segment_inputs:
        _df_seg = df[(df["elapsed_sec"] >= _start_sec) & (df["elapsed_sec"] <= _end_sec)]
        _hr_seg = _df_seg["heart_rate"].dropna()
        if len(_hr_seg) > 1:
            st.plotly_chart(
                build_density_chart(
                    _hr_seg,
                    f"Heart Rate Density Distribution - {_seg_name}",
                    _avg_hr_total,
                    z1, z2, z3, z4, z5,
                    _zone_bands,
                    x_min=_x_min_total, x_max=_x_max_total
                ),
                use_container_width=True
            )
        else:
            st.warning(f"⚠️ Not enough HR data for segment {_seg_name}")

# =====================================
# 🌡️ HEATMAP TIME-IN-ZONE IN MINUTES
# =====================================

if uploaded_file is not None and bar_df is not None and not bar_df.empty:

    # Funzione sicura per convertire H:MM in minuti
    def h_mm_to_minutes(hmm_str):
        try:
            h, m = map(int, str(hmm_str).split(":"))
            return h*60 + m
        except:
            return 0

    # Crea copia del DataFrame per sicurezza
    heatmap_df_minutes = bar_df.copy()

    # Applica la conversione a ogni cella di ogni colonna
    for col in heatmap_df_minutes.columns:
        heatmap_df_minutes[col] = heatmap_df_minutes[col].apply(h_mm_to_minutes)

    # Imposta l'indice con i nomi delle zone
    heatmap_df_minutes.index = [f"Zone {i+1}" for i in range(len(heatmap_df_minutes))]

    # Reset index e reshape per Plotly
    heatmap_df_minutes_reset = heatmap_df_minutes.reset_index().rename(columns={'index':'HR Zone'})
    heatmap_long = heatmap_df_minutes_reset.melt(
        id_vars="HR Zone",
        var_name="Segment",
        value_name="Minutes"
    )

    # Debug (opzionale) per verificare i dati
    # st.write("DEBUG heatmap_long:", heatmap_long.head())

    # Creazione heatmap con Plotly
    fig_heat = px.density_heatmap(
        heatmap_long,
        x="Segment",
        y="HR Zone",
        z="Minutes",
        text_auto=True,
        color_continuous_scale="YlOrRd",
        hover_data={"Segment": True, "HR Zone": True, "Minutes": True},
        title="🌡️ Time-in-Zone (minutes) Heatmap"
    )

    # Inverti l'asse y per convenzione zone
    fig_heat.update_layout(
        yaxis=dict(autorange='reversed'),
        coloraxis_colorbar=dict(title="Time (minutes)")
    )

    # Mostra heatmap su Streamlit
    st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.warning(" ⚠️ Please submit all data required for the analysis.")

# -----------------------------
# LAP / CLIMB ANALYSIS
# -----------------------------

if 'df' not in locals() or df.empty:
    st.warning("⚠️ Please upload a FIT file first to perform analysis.")

elif 'HR Zone' not in df.columns:
    st.warning("⚠️ Please submit the Heart Rate Zones first to enable Lap/Climb analysis.")

else:
    hr_zone_map = {
        "Zone 1 // Aerobic Low": "Z1",
        "Zone 2 // Aerobic High": "Z2",
        "Zone 3 // Aerobic Endurance": "Z3",
        "Zone 4 // Sub Threshold": "Z4",
        "Zone 5 // Super Threshold": "Z5"
    }
    zone_order = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    pct_cols   = [f"% {z}" for z in zone_order]

    colors = [
        "royalblue", "tomato", "gold", "mediumseagreen", "orchid",
        "darkorange", "deepskyblue", "limegreen", "crimson", "slateblue"
    ]

    # --- Helpers ---

    def parse_time_to_seconds(t):
        if t is None:
            return 0
        if isinstance(t, (int, float)):
            return int(t)
        t = str(t).strip()
        if t == "":
            return 0
        parts = t.split(":")
        try:
            if len(parts) == 2:
                h, m = map(int, parts)
                return h * 3600 + m * 60
            elif len(parts) == 3:
                h, m, s = map(int, parts)
                return h * 3600 + m * 60 + s
            elif len(parts) == 1:
                return int(float(parts[0]))
            return 0
        except:
            return 0

    def format_hms(sec):
        sec = int(sec)
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h}:{m:02d}:{s:02d}"

    def format_hm(sec):
        sec = int(sec)
        h = sec // 3600
        m = (sec % 3600) // 60
        return f"{h}:{m:02d}"

    def get_segment_df(entry):
        """Slice the main df for a given lap/climb entry."""
        if "start_idx" in entry and "end_idx" in entry:
            return df.loc[entry["start_idx"]:entry["end_idx"]].copy()
        start_s = parse_time_to_seconds(entry.get("start_time", "0:00"))
        end_s   = parse_time_to_seconds(entry.get("end_time",   "0:01"))
        return df[(df["elapsed_sec"] >= start_s) & (df["elapsed_sec"] <= end_s)].copy()

    def build_analysis_df(data, mode):
        rows = []

        for entry in data:
            df_seg = get_segment_df(entry)
            if df_seg.empty:
                continue

            duration_sec = max(int(df_seg["elapsed_sec"].max() - df_seg["elapsed_sec"].min()), 1)
            duration_hms = format_hms(duration_sec)

            df_seg["time_diff_sec"] = df_seg["elapsed_sec"].diff().fillna(0)
            df_seg["HR Zone Short"] = df_seg["HR Zone"].map(hr_zone_map)

            lap_summary = (
                df_seg.groupby("HR Zone Short")["time_diff_sec"]
                .sum()
                .reindex(zone_order)
                .fillna(0)
            )
            lap_summary_hm = [format_hm(x) for x in lap_summary.values]
            pct_zones      = [f"{round((x / duration_sec) * 100)}%" for x in lap_summary.values]

            avg_fc    = int(df_seg["heart_rate"].mean()) if "heart_rate" in df_seg.columns else 0
            distance  = round(df_seg["distance_km"].max() - df_seg["distance_km"].min(), 1)
            elevation = int(df_seg["elevation_m"].diff().clip(lower=0).sum())
            ngp       = entry.get("ngp", "")

            if mode == "climb":
                avg_grade = round((elevation / distance / 10) if distance > 0 else 0)
                vam       = round(elevation / (duration_sec / 3600) if duration_sec > 0 else 0)
                rows.append([
                    entry.get("name", "Climb"),
                    duration_hms, distance, elevation,
                    avg_fc, avg_grade, vam, ngp
                ] + lap_summary_hm + pct_zones)

            else:
                if distance > 0:
                    pace_total = duration_sec / distance
                    pace_min   = int(pace_total // 60)
                    pace_sec   = int(round(pace_total - pace_min * 60))
                    if pace_sec == 60:
                        pace_min += 1
                        pace_sec  = 0
                    lap_pace = f"{pace_min:02d}:{pace_sec:02d}"
                else:
                    lap_pace = "00:00"

                rows.append([
                    entry.get("name", "Lap"),
                    duration_hms, distance, elevation,
                    avg_fc, lap_pace, ngp
                ] + lap_summary_hm + pct_zones)

        if not rows:
            return None, None

        if mode == "climb":
            extra_cols = ["Avg FC", "Avg Grade (%)", "VAM (m/h)", "NGP"]
            name_col   = "Climb Name"
        else:
            extra_cols = ["Avg FC", "Lap Pace (min/km)", "NGP"]
            name_col   = "Lap Name"

        columns   = [name_col, "Duration", "Distance (km)", "Elevation (m)"] + extra_cols + zone_order + pct_cols
        result_df = pd.DataFrame(rows, columns=columns)
        result_df["Distance (km)"] = result_df["Distance (km)"].astype(float).map("{:.1f}".format)
        return result_df, name_col

    def show_elevation_profile(data, mode):
        """Elevation profile with each lap/climb highlighted in a different color."""
        # Laps use distance on x-axis; climbs use elapsed time
        if mode == "lap":
            x_base  = df["distance_km"]
            x_label = "Distance (km)"
        else:
            x_base  = df["elapsed_sec"].apply(seconds_to_hhmm)
            x_label = "Elapsed Time (HH:MM)"

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_base,
            y=df["elevation_m"].astype(int),
            mode="lines",
            line=dict(color="gray", width=1.5),
            name="Elevation",
            hovertemplate=f"{x_label}: %{{x}}<br>Elevation: %{{y}} m<extra></extra>"
        ))

        for i, entry in enumerate(data):
            df_seg = get_segment_df(entry)
            if df_seg.empty:
                continue

            color = colors[i % len(colors)]
            x_seg = df_seg["distance_km"] if mode == "lap" else df_seg["elapsed_sec"].apply(seconds_to_hhmm)

            fig.add_trace(go.Scatter(
                x=x_seg,
                y=df_seg["elevation_m"].astype(int),
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=color,
                opacity=0.35,
                name=entry.get("name", f"Segment {i+1}"),
                hovertemplate=f"{entry.get('name', '')}<br>{x_label}: %{{x}}<br>Elevation: %{{y}} m<extra></extra>"
            ))

        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title="Elevation (m)",
            hovermode="x unified",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=80)
        )
        fig.update_yaxes(tickformat="d")
        st.plotly_chart(fig, use_container_width=True)

    def show_zone_bar_chart(result_df, name_col, title):
        """Grouped bar chart: % time in HR zones per lap/climb."""
        df_plot = result_df[[name_col] + pct_cols].copy()
        for col in pct_cols:
            df_plot[col] = df_plot[col].str.rstrip('%').astype(float)
        df_melted = df_plot.melt(
            id_vars=[name_col],
            value_vars=pct_cols,
            var_name="HR Zone",
            value_name="Percentage"
        )
        fig = px.bar(
            df_melted,
            x="HR Zone", y="Percentage",
            color=name_col, barmode="group",
            title=title
        )
        st.plotly_chart(fig, use_container_width=True)

    # ===========================================================
    # ⛰️ CLIMB ANALYSIS
    # ===========================================================
    if st.session_state.get("do_climb_analysis") and st.session_state.get("climb_data"):
        st.markdown("---")
        st.markdown("### ⛰️ Climb Analysis")

        climb_df, climb_name_col = build_analysis_df(st.session_state["climb_data"], mode="climb")

        if climb_df is not None:
            st.session_state["climb_zone_df"] = climb_df

            show_elevation_profile(st.session_state["climb_data"], mode="climb")

            edited_climb_df = st.data_editor(
                climb_df,
                num_rows="dynamic",
                use_container_width=True,
                disabled=True
            )
            st.session_state["climb_zone_data_edited"] = edited_climb_df

            show_zone_bar_chart(climb_df, climb_name_col, "⛰️ Climb — % Time in HR Zones")

        else:
            st.warning("⚠️ No climb data could be computed. Check your climb definitions.")

    elif st.session_state.get("do_climb_analysis"):
        st.info("👆 No climb data yet — define your climbs in the section above.")

    # ===========================================================
    # 🏃 LAP ANALYSIS
    # ===========================================================
    if st.session_state.get("do_lap_analysis") and st.session_state.get("lap_data"):
        st.markdown("---")
        st.markdown("### 🏃 Lap Analysis")

        lap_df, lap_name_col = build_analysis_df(st.session_state["lap_data"], mode="lap")

        if lap_df is not None:
            st.session_state["lap_zone_df"] = lap_df

            show_elevation_profile(st.session_state["lap_data"], mode="lap")

            edited_lap_df = st.data_editor(
                lap_df,
                num_rows="dynamic",
                use_container_width=True,
                disabled=True
            )
            st.session_state["lap_zone_data_edited"] = edited_lap_df

            show_zone_bar_chart(lap_df, lap_name_col, "🏃 Lap — % Time in HR Zones")

        else:
            st.warning("⚠️ No lap data could be computed. Check your lap definitions.")

    elif st.session_state.get("do_lap_analysis"):
        st.info("👆 No lap data yet — define your laps in the section above.")
### COACH COMMENT SECTION ###

st.subheader("Coach Comment Section")

# Initialize the comment in session state
if "comment" not in st.session_state:
    st.session_state.comment = ""

# Text input for the user
comment_input = st.text_area("Insert your comment here:", value=st.session_state.comment)

# Save button
if st.button("Save Comment"):
    st.session_state.comment = comment_input
    st.success("Comment saved!")

# Display the stored comment
if st.session_state.comment:
    st.subheader("Coach Comment:")
    st.write(st.session_state.comment)


# --------------------------------------------------------------------
# PDF GENERATION
# --------------------------------------------------------------------
class ModernPDF(FPDF):
    def header(self):
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 30, 'F')
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "DU COACHING - Race Analyzer Report", ln=True)
        self.set_xy(10, 18)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(200, 200, 200)
        self.cell(0, 6, "This analyzer is brought to you by Coach Ambro", ln=True)
        self.ln(5)

    def section_title(self, title):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)

    def body_text(self, text):
        self.set_font("Helvetica", "", 11)
        self.set_text_color(55, 55, 55)
        self.multi_cell(0, 6, text)
        self.ln(2)

    def add_spacer(self, h=4):
        self.ln(h)

def build_density_chart_matplotlib(hr_data, title, avg_hr, z1, z2, z3, z4, z5, x_min=None, x_max=None):
    kde = gaussian_kde(hr_data, bw_method=0.3)
    x_range = np.linspace(hr_data.min(), hr_data.max(), 500)
    y_kde = kde(x_range)
    y_kde = (y_kde / y_kde.sum()) * 100

    if x_min is None:
        x_min = np.percentile(hr_data, 1)
    if x_max is None:
        x_max = hr_data.max()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_kde.max() * 1.25)

    zone_bands = [
        {"name": "Z1", "x0": 0,   "x1": z1, "color": (0.39, 0.78, 1.0,  0.2)},
        {"name": "Z2", "x0": z1,  "x1": z2, "color": (0.39, 0.86, 0.39, 0.2)},
        {"name": "Z3", "x0": z2,  "x1": z3, "color": (1.0,  0.90, 0.20, 0.2)},
        {"name": "Z4", "x0": z3,  "x1": z4, "color": (1.0,  0.59, 0.20, 0.2)},
        {"name": "Z5", "x0": z4,  "x1": z5, "color": (1.0,  0.31, 0.31, 0.2)},
    ]

    for zone in zone_bands:
        band_x0 = max(zone["x0"], x_min)
        band_x1 = min(zone["x1"], x_max)
        if band_x0 >= band_x1:
            continue
        ax.axvspan(band_x0, band_x1, color=zone["color"])
        mid = (band_x0 + band_x1) / 2
        ax.text(mid, y_kde.max() * 1.15, zone["name"],
                ha="center", va="center", fontsize=10, fontweight="bold", color="dimgray")

    ax.fill_between(x_range, y_kde, alpha=0.3, color="steelblue")
    ax.plot(x_range, y_kde, color="steelblue", linewidth=2)

    for bpm in [z1, z2, z3, z4, z5]:
        if x_min <= bpm <= x_max:
            ax.axvline(x=bpm, color="gray", linestyle="--", linewidth=1)

    ax.axvline(x=avg_hr, color="crimson", linestyle="-", linewidth=2.5)
    ax.text(
        avg_hr, y_kde.max() * 1.10,
        f"AVG\n{avg_hr:.0f} bpm",
        ha="left", va="center",
        fontsize=9, color="crimson",
        bbox=dict(facecolor="white", edgecolor="crimson", alpha=0.7, boxstyle="round,pad=0.3")
    )

    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig

def add_chart_to_pdf(fig, title=None):
    if title:
        pdf.section_title(title)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="PNG", dpi=200, bbox_inches="tight")
        tmpfile.close()
        pdf.image(tmpfile.name, x=10, w=190)
    plt.close(fig)

# --- Reusable helper: elevation profile chart for PDF ---
def pdf_elevation_with_segments(data, mode, title):
    fig, ax = plt.subplots(figsize=(10, 4))

    if mode == "lap":
        x_base  = df["distance_km"]
        x_label = "Distance (km)"
    else:
        x_base  = df["elapsed_sec"]
        x_label = "Elapsed Time (hh:mm)"

    ax.plot(x_base, df["elevation_m"], color="gray", label="Elevation")

    tab_colors = plt.cm.tab10.colors
    for idx, c in enumerate(data):
        color = tab_colors[idx % len(tab_colors)]

        if "start_idx" in c and "end_idx" in c:
            s, e  = int(c["start_idx"]), int(c["end_idx"])
            x_seg = x_base.iloc[s:e+1]
            y_seg = df["elevation_m"].iloc[s:e+1]
        else:
            st_sec  = hhmm_to_seconds(c.get("start_time", "0:00"))
            end_sec = hhmm_to_seconds(c.get("end_time",   "0:01"))
            if st_sec is None or end_sec is None:
                continue
            if mode == "lap":
                mask  = (df["distance_km"] >= c.get("start_km", 0)) & (df["distance_km"] <= c.get("end_km", 0))
            else:
                mask  = (df["elapsed_sec"] >= st_sec) & (df["elapsed_sec"] <= end_sec)
            x_seg = x_base[mask]
            y_seg = df.loc[mask, "elevation_m"]

        if x_seg.empty:
            continue

        ax.fill_between(x_seg, y_seg, alpha=0.3, color=color, label=c.get("name", f"Segment {idx+1}"))
        ax.plot(x_seg, y_seg, color=color, linewidth=2)

    if mode != "lap":
        def sec_to_hhmm_formatter(x, pos):
            h = int(x // 3600)
            m = int((x % 3600) // 60)
            return f"{h:02d}:{m:02d}"
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmm_formatter))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_xlabel(x_label)
    ax.set_ylabel("Elevation (m)")
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    return fig

# --- Reusable helper: % time in zone bar chart for PDF ---
def pdf_zone_bar_chart(zone_df, name_col, title):
    zones     = ["Z1", "Z2", "Z3", "Z4", "Z5"]
    n_entries = len(zone_df)
    x         = np.arange(len(zones))
    bar_width = 0.8 / max(n_entries, 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (_, row) in enumerate(zone_df.iterrows()):
        pct_values = []
        for z in zones:
            val = row.get(f"% {z}", "0")
            if isinstance(val, str):
                val = val.rstrip('%')
            pct_values.append(float(val or 0))
        ax.bar(x + i * bar_width, pct_values, width=bar_width,
               label=row.get(name_col, f"Entry {i+1}"))

    ax.set_xticks(x + bar_width * (n_entries - 1) / 2)
    ax.set_xticklabels(zones)
    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

# --- Reusable helper: write a lap/climb info + HR table to PDF ---
def pdf_write_analysis_table(zone_df, section_label, name_col):
    zone_df_copy = zone_df.copy()

    # Info table (no Z/% columns)
    info_cols = [col for col in zone_df_copy.columns if not col.startswith("Z") and not col.startswith("%")]
    n_cols    = len(info_cols)
    col_width = (pdf.w - pdf.l_margin - pdf.r_margin) / n_cols
    row_height = 6

    pdf.section_title(f"{section_label} - Info Table")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(20, 20, 20)
    for col in info_cols:
        pdf.cell(col_width, row_height, str(col)[:12], border=1, fill=True, align='C')
    pdf.ln(row_height)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    for _, row in zone_df_copy.iterrows():
        for col in info_cols:
            pdf.cell(col_width, row_height, str(row[col]), border=1, align='C')
        pdf.ln(row_height)
    pdf.add_spacer(6)

    # HR zone columns table
    hr_cols   = [name_col] + [f"Z{i}" for i in range(1, 6)] + [f"% Z{i}" for i in range(1, 6)]
    n_cols    = len(hr_cols)
    col_width = (pdf.w - pdf.l_margin - pdf.r_margin) / n_cols

    pdf.section_title(f"{section_label} - HR Data Analysis")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(20, 20, 20)
    for col in hr_cols:
        pdf.cell(col_width, row_height, col, border=1, fill=True, align='C')
    pdf.ln(row_height)

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(60, 60, 60)
    for _, row in zone_df_copy.iterrows():
        pdf.cell(col_width, row_height, str(row[name_col]), border=1, align='C')
        for i in range(1, 6):
            pdf.cell(col_width, row_height, str(row.get(f"Z{i}", "")), border=1, align='C')
        for i in range(1, 6):
            pdf.cell(col_width, row_height, str(row.get(f"% Z{i}", "")), border=1, align='C')
        pdf.ln(row_height)
    pdf.add_spacer(6)


# --------------------------------------------------------------------
# PDF REPORT GENERATION LOGIC
# --------------------------------------------------------------------
if uploaded_file is not None and 'df' in locals() and not df.empty and 'HR Zone' in df.columns and 'athlete_name' in st.session_state:
    if st.button("📄 Generate PDF Report"):

        pdf = ModernPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- Athlete & Race Info ---
        pdf.section_title("Athlete & Race Information")
        pdf.body_text(f"Athlete: {athlete_name}")
        pdf.body_text(f"Race: {race_name}")
        pdf.body_text(f"Date: {formatted_date}")
        pdf.body_text(f"Distance: {kilometers} km")
        pdf.body_text(f"Elevation: {total_elevation_gain} m")
        pdf.body_text(f"Final Time: {final_time_str}")
        pdf.add_spacer()
        pdf.body_text(f"Overall Average HR: {overall_avg:.0f} bpm")
        pdf.body_text(f"First Half Avg HR: {first_half_avg:.0f} bpm")
        pdf.body_text(f"Second Half Avg HR: {second_half_avg:.0f} bpm")
        pdf.body_text(f"% Difference: {percent_diff:.1f}%")
        pdf.body_text(f"DET Index: {det_index_str} ({comment})")

        pdf.add_page()

        # --- Elevation Profile with Time Segments ---
        if 'df' in locals() and not df.empty:
            fig2, ax2 = plt.subplots(figsize=(10, 4))

            ax2.plot(df["elapsed_sec"], df["elevation_m"], color="gray", label="Elevation")

            segment_colors = ["royalblue", "tomato", "gold", "mediumseagreen", "orchid",
                              "darkorange", "deepskyblue", "limegreen", "crimson", "slateblue"]

            for i in range(1, st.session_state.get('num_segments', 1) + 1):
                start_key = f'segment{i}_start'
                end_key   = f'segment{i}_end'
                if all(k in st.session_state for k in [start_key, end_key]):
                    seg_start = h_mm_to_seconds(st.session_state[start_key])
                    seg_end   = h_mm_to_seconds(st.session_state[end_key])
                    if seg_start is not None and seg_end is not None and seg_end > seg_start:
                        mask      = (df["elapsed_sec"] >= seg_start) & (df["elapsed_sec"] <= seg_end)
                        seg_label = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                        ax2.fill_between(
                            df["elapsed_sec"], df["elevation_m"],
                            where=mask, alpha=0.25,
                            color=segment_colors[(i-1) % len(segment_colors)],
                            label=seg_label
                        )

            def sec_to_hhmm_formatter(x, pos):
                h = int(x // 3600)
                m = int((x % 3600) // 60)
                return f"{h:02d}:{m:02d}"

            ax2.xaxis.set_major_formatter(mticker.FuncFormatter(sec_to_hhmm_formatter))
            ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
            plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
            ax2.set_xlabel("Elapsed Time (hh:mm)")
            ax2.set_ylabel("Elevation (m)")
            ax2.set_title("Elevation Profile with Time Segments Highlighted")
            ax2.grid(True)
            ax2.legend(loc="upper left", fontsize=8)
            plt.tight_layout()
            add_chart_to_pdf(fig2, title="Elevation Profile - Time Segments")

        # --- Time-in-Zone Table ---
        if "combined_df" in locals():
            pdf.section_title("Time-in-Zone Table [hh:mm]")
            page_w    = pdf.w - pdf.l_margin - pdf.r_margin
            n_cols    = 1 + len(combined_df.columns)
            col_width = page_w / n_cols
            row_height = 8

            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(col_width, row_height, "HR Zone", border=1, fill=True)
            for col in combined_df.columns:
                pdf.cell(col_width, row_height, str(col), border=1, fill=True)
            pdf.ln()

            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            for i, zone in enumerate(combined_df.index, 1):
                pdf.cell(col_width, row_height, f"Zone {i}", border=1)
                for seg in combined_df.columns:
                    pdf.cell(col_width, row_height, str(combined_df.loc[zone, seg]), border=1)
                pdf.ln()
            pdf.add_spacer(1)

        # --- Bar Chart & Heatmap ---
        if "bar_df" in locals():
            fig, ax = plt.subplots(figsize=(10, 4))
            zones = np.arange(len(bar_df.index))
            width = 0.2

            for i, seg in enumerate(bar_df.columns):
                vals = bar_df[seg].apply(lambda t: int(t.split(':')[0]) + int(t.split(':')[1])/60)
                ax.bar(zones + i * width, vals, width=width, label=seg)

            ax.set_xticks(zones + width * (len(bar_df.columns)-1)/2)
            ax.set_xticklabels(bar_df.index)
            ax.set_ylabel("Hours")
            ax.set_title("Time-in-Zone per Segment")
            ax.legend(title="Segment")
            plt.tight_layout()
            add_chart_to_pdf(fig, title="Time-in-Zone - Bar Chart")
            pdf.add_page()

        if "heatmap_df_minutes" in locals() and not heatmap_df_minutes.empty:
            heatmap_numeric = heatmap_df_minutes.copy()
            for col in heatmap_numeric.columns:
                heatmap_numeric[col] = pd.to_numeric(heatmap_numeric[col], errors='coerce').fillna(0)

            fig, ax = plt.subplots(figsize=(10, 3 + max(0, len(heatmap_numeric)/4)))
            sns.heatmap(
                heatmap_numeric, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Minutes'}, ax=ax
            )
            ax.set_title("Time-in-Zone Heatmap (Minutes)")
            ax.set_ylabel("HR Zone")
            ax.set_xlabel("Segment")
            plt.tight_layout()
            add_chart_to_pdf(fig, title="Time-in-Zone - Heatmap")
            pdf.add_page()

        # ===========================================================
        # ⛰️ CLIMB ANALYSIS IN PDF
        # ===========================================================
        if st.session_state.get("do_climb_analysis") and st.session_state.get("climb_data"):
            climb_data   = st.session_state["climb_data"]
            climb_zone_df = st.session_state.get("climb_zone_df")

            fig = pdf_elevation_with_segments(climb_data, mode="climb", title="Elevation Profile - Climbs Highlighted")
            add_chart_to_pdf(fig, title="Climb Analysis")

            if climb_zone_df is not None and not climb_zone_df.empty:
                pdf_write_analysis_table(climb_zone_df, "Climb Analysis", "Climb Name")
                fig = pdf_zone_bar_chart(climb_zone_df, "Climb Name", "Climb — % Time in HR Zones")
                add_chart_to_pdf(fig)
                pdf.add_page()

        # ===========================================================
        # 🏃 LAP ANALYSIS IN PDF
        # ===========================================================
        if st.session_state.get("do_lap_analysis") and st.session_state.get("lap_data"):
            lap_data    = st.session_state["lap_data"]
            lap_zone_df = st.session_state.get("lap_zone_df")

            fig = pdf_elevation_with_segments(lap_data, mode="lap", title="Elevation Profile - Laps Highlighted")
            add_chart_to_pdf(fig, title="Lap Analysis")

            if lap_zone_df is not None and not lap_zone_df.empty:
                pdf_write_analysis_table(lap_zone_df, "Lap Analysis", "Lap Name")
                fig = pdf_zone_bar_chart(lap_zone_df, "Lap Name", "Lap — % Time in HR Zones")
                add_chart_to_pdf(fig)
                pdf.add_page()

        # --- HR Trend Chart ---
        fig = plt.figure(figsize=(10, 4))
        plt.plot(df["elapsed_hours"], df["hr_smooth"], label="HR Smooth")
        try:
            plt.plot(df["elapsed_hours"], reg.predict(X), label="Trend Line", linestyle="--")
        except Exception:
            pass
        plt.xlabel("Elapsed Time (hours)")
        plt.ylabel("Heart Rate (bpm)")
        plt.title("Heart Rate Over Time")
        plt.legend()
        plt.tight_layout()
        add_chart_to_pdf(fig, title="Heart Rate - Trend Analysis")
        pdf.body_text(f"DET Index: {det_index_str} ({comment})")
        pdf.add_spacer(4)

        # --- HR Density Distribution Charts ---
        pdf.add_page()
        hr_data_total = df["heart_rate"].dropna()
        x_min_total   = np.percentile(hr_data_total, 1)
        x_max_total   = hr_data_total.max()

        if len(hr_data_total) > 1:
            fig = build_density_chart_matplotlib(
                hr_data_total, "HR Density - Full Race", overall_avg, z1, z2, z3, z4, z5
            )
            add_chart_to_pdf(fig, title="Heart Rate Density Distribution")

        for i in range(1, st.session_state.get('num_segments', 1) + 1):
            start_key = f'segment{i}_start'
            end_key   = f'segment{i}_end'
            if all(k in st.session_state for k in [start_key, end_key]):
                _pdf_start_sec = h_mm_to_seconds(st.session_state[start_key])
                _pdf_end_sec   = h_mm_to_seconds(st.session_state[end_key])
                if _pdf_start_sec is not None and _pdf_end_sec is not None and _pdf_end_sec > _pdf_start_sec:
                    seg_name = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                    df_seg   = df[(df["elapsed_sec"] >= _pdf_start_sec) & (df["elapsed_sec"] <= _pdf_end_sec)]
                    hr_seg   = df_seg["heart_rate"].dropna()
                    if len(hr_seg) > 1:
                        fig = build_density_chart_matplotlib(
                            hr_seg, f"HR Density - {seg_name}", overall_avg, z1, z2, z3, z4, z5,
                            x_min=x_min_total, x_max=x_max_total
                        )
                        add_chart_to_pdf(fig)

        # --- Coach Comment ---
        if st.session_state.comment:
            pdf.section_title("Coach comment")
            pdf.body_text(f"{st.session_state.comment}")

        pdf_data = pdf.output(dest="S")
        if isinstance(pdf_data, str):
            pdf_bytes = pdf_data.encode('latin1')
        elif isinstance(pdf_data, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_data)
        else:
            raise TypeError(f"Unexpected type from FPDF output: {type(pdf_data)}")

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state['athlete_name']}_{st.session_state['race_name']}_{st.session_state['race_date'].year}_Race_report.pdf",
            mime="application/pdf"
        )
else:
    st.warning("⚠️ Please submit race, athlete and cardiac data to generate the PDF report")