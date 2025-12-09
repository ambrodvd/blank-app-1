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
        df["distance_km"] = df["distance_km"].fillna(method="ffill").fillna(0).astype(float)

        df["elevation_m"] = df["enhanced_altitude"].fillna(method="ffill").fillna(0).astype(float)

        df["lat"] = df["position_lat"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)
        df["lon"] = df["position_long"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)

        # Elapsed time safely
        if "timestamp" in df.columns and not df["timestamp"].isna().all():
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            start_time = df["timestamp"].iloc[0]
            df["elapsed_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
        else:
            df["elapsed_sec"] = np.arange(len(df))

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
    z1 = st.number_input("Zone 1 (Recovery) - up to:", min_value=60, value=st.session_state['z1'])
    z2 = st.number_input("Zone 2 (Aerobic) - up to:", min_value=60, value=st.session_state['z2'])
    z3 = st.number_input("Zone 3 (Tempo) - up to:", min_value=60, value=st.session_state['z3'])
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
        - 🩵 Zone 1 (Recovery): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Tempo): {z2+1} - {z3} bpm  
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

# --- Time Segment Input Form (Start → End) ---
with st.form("time_segment_form"):
    st.subheader("⏱️ Time segments for Time in Zone Analysis")
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
            df[col] = df[col].fillna(method="ffill").fillna(0)     # fill NaNs

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

# -----------------------------------------------------------
# 1️⃣ SELECT ANALYSIS TYPE (LAP / CLIMB / NONE)
# -----------------------------------------------------------
analysis_type = st.radio(
    "Which type of analysis would you like to perform?",
    ("None", "Lap Analysis", "Climb Analysis"),
    index=0,
    key="analysis_type_selector"
)

# Store clean value in session_state
if analysis_type == "Lap Analysis":
    st.session_state["analysis_type"] = "lap"
    st.session_state["do_lap_analysis"] = True
elif analysis_type == "Climb Analysis":
    st.session_state["analysis_type"] = "climb"
    st.session_state["do_lap_analysis"] = True
else:
    st.session_state["analysis_type"] = None
    st.session_state["do_lap_analysis"] = False

if analysis_type != "Climb Analysis":
    st.session_state["climb_data_insert"] = None
#------------#
#----------- CLIMB ANALYS CHOICHE --------#

if st.session_state.get("do_lap_analysis") and st.session_state.get("analysis_type") == "climb":
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
# CLIMB ANALYSIS WORKFLOW (FRAGMENT + PLOTLY EDITABLE TABLE)
# ---------------------------

if (
    st.session_state.get("analysis_type") == "climb"
    and st.session_state.get("climb_data_insert") == "automatic"
    and "fit_df" in st.session_state
):

    df = st.session_state["fit_df"]

    # --- Detection parameters ---
    min_elev_gain = st.number_input("Minimum elevation gain for a climb (meters)", value=300)
    max_downhill = st.number_input("Maximum allowed downhill inside a climb (meters)", value=5)

    # --- Step 1: Detect climbs (cached) ---
    @st.cache_data
    def cached_detect_climbs(df, min_elev_gain, max_downhill):
        return detect_climbs(df, min_elev_gain=min_elev_gain, max_downhill=max_downhill)

    raw_climbs = cached_detect_climbs(df, min_elev_gain, max_downhill)
    if not raw_climbs:
        st.warning("No climbs detected.")
        st.stop()

    temp_climbs = add_temporary_metrics(df, [c.copy() for c in raw_climbs])

    # --- Step 2: Lightweight preview plot ---
    st.subheader("Automatically detected climbs. You can edit them in the table")
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

    # --- Step 3: Editable table fragment ---
    @st.fragment
    def climb_table_fragment():
        st.subheader("Detected Climbs")
        st.info("You can edit the climb names, start times and end times")

        # Initialize table once
        if "editable_climb_table" not in st.session_state:
            editable_df = pd.DataFrame(temp_climbs)
            cols_desired = ["Climb Name", "start_time", "end_time", "distance_km", "elev_gain_m"]
            for col in cols_desired:
                if col not in editable_df.columns:
                    editable_df[col] = ""
            editable_df = editable_df[cols_desired]
            st.session_state["editable_climb_table"] = editable_df

        # Load current table
        editable_df = st.session_state["editable_climb_table"]

        # Show editable table
        edited = st.data_editor(
            editable_df,
            num_rows="dynamic",
            key="climb_editor",
            disabled=["distance_km", "elev_gain_m"]
        )

        # Save edits in session_state (won't compute anything yet)
        st.session_state["editable_climb_table"] = edited

    climb_table_fragment()

    # --- Step 4: Save button ---
    if st.button("Save data and compute final metrics"):
        try:
            processed_climbs = compute_processed_climbs(df, st.session_state["editable_climb_table"])
            st.session_state["climb_data"] = processed_climbs
            st.session_state["lap_data"] = processed_climbs
            st.success("✅ Climbs saved and metrics computed!")
        except ValueError as e:
            st.error(str(e))

    # --- Step 5: Final plot + table after SAVE ---
    if "climb_data" in st.session_state:
        final_climbs = st.session_state["climb_data"]

        x_axis_option_final = st.radio("X-axis for climb plot:", ["Elapsed Time", "Distance (km)"], key="xaxis_final")
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
            title="Selected climbs highlighted below",
            hovermode="x unified",
            showlegend=False
        )
        fig_final.update_yaxes(title_text="Elevation (m)", tickformat="d")
        st.plotly_chart(fig_final, use_container_width=True)

        final_df = pd.DataFrame(final_climbs)
        display_cols = ["name", "start_time", "end_time", "duration", "distance", "elevation"]
        st.dataframe(final_df[[c for c in display_cols if c in final_df.columns]].reset_index(drop=True))

# ---------------------------
# LAP ANALYSIS WITH AUTO LOOP DETECTOR
# ---------------------------
if st.session_state.get("do_lap_analysis") and st.session_state.get("analysis_type") == "lap":
    lap_data_insert = st.radio(
        "How do you want to insert lap data?",
        ("Manually", "Automatic Lap detector", "Distance slicer"),
        index=0,
        key="lap_data_insert_selector"
    )

    # Store clean value
    if lap_data_insert == "Manually":
        st.session_state["lap_data_insert"] = "manual"
    elif lap_data_insert == "Automatic Lap detector":
        st.session_state["lap_data_insert"] = "automatic"
    else:
        st.session_state["lap_data_insert"] = "fix_distance"

# ---------------------------
# AUTOMATIC LOOP DETECTOR
# ---------------------------

loops = []  # default empty

if st.session_state.get("lap_data_insert") == "automatic":
    if uploaded_file is not None and "df" in locals():
        # --- Parameters ---
        col1, col2, col3 = st.columns(3)
        with col1:
            base_radius = st.number_input("Base radius (meters)", value=20, min_value=1)
        with col2:
            percent_error = st.slider("GPS antenna error (%)", 0.0, 5.0, 1.0)/100
        with col3:
            min_samples_between_crossings = st.slider("Min samples between crossings", 1, 50, 5)

        # --- Validate required columns ---
        required_cols = ["lat", "lon", "timestamp", "elevation_m", "distance_km", "elapsed_sec"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Missing required columns: {', '.join([c for c in required_cols if c not in df.columns])}")
            st.stop()

        df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)

        # --- Ensure numeric types ---
        df["elevation_m"] = df["elevation_m"].astype(float)
        df["distance_km"] = df["distance_km"].astype(float)

        # --- Compute distance from start (meters) for loop detection ---
        from haversine import haversine

        start_lat, start_lon = df.loc[0, ["lat", "lon"]]
        df["dist_from_start_m"] = df.apply(
            lambda r: haversine((start_lat, start_lon), (r["lat"], r["lon"])) * 1000,
            axis=1
        )
        threshold = base_radius + percent_error * df["dist_from_start_m"].max()

        # --- Loop detection ---
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

        # --- Build loops ---
        if len(crossings) > 1:
            loops = []
            for i in range(1, len(crossings)):
                prev_idx, _ = crossings[i-1]
                idx, _ = crossings[i]

                df_lap = df.iloc[prev_idx:idx+1].copy()

                # Duration
                start_sec = df_lap["elapsed_sec"].iloc[0]
                end_sec = df_lap["elapsed_sec"].iloc[-1]
                duration_sec = end_sec - start_sec
                duration_hhmm = seconds_to_hhmm(duration_sec)

                # Distance in km
                lap_distance = df_lap["distance_km"].max() - df_lap["distance_km"].min()

                # Elevation gain in meters
                lap_elevation = df_lap["elevation_m"].diff().clip(lower=0).sum()

                # Save the loop
                loops.append({
                    "name": f"Lap {i}",
                    "start_time": seconds_to_hhmm(start_sec),
                    "end_time": seconds_to_hhmm(end_sec),
                    "duration": duration_hhmm,
                    "start_idx": prev_idx,
                    "end_idx": idx,
                    "distance": lap_distance,
                    "elevation": lap_elevation,
                    "ngp": ""
                })

        st.session_state["lap_data"] = loops
        st.session_state["lap_form_submitted"] = True

        st.subheader("Detected Loops")
        if loops:
            loops_df = pd.DataFrame(loops)

            # --- Format distance and elevation ---
            loops_df["distance"] = loops_df["distance"].astype(float).map("{:.1f}".format)
            loops_df["elevation"] = loops_df["elevation"].astype(int)

            st.dataframe(loops_df[["name", "start_time", "end_time", "duration", "distance", "elevation"]])
        else:
            st.warning("No loops detected.")


# ---------------------------
# FIX DISTANCE / DISTANCE SLICER
# ---------------------------
if st.session_state.get("lap_data_insert") == "fix_distance":
    if uploaded_file is not None and 'df' in locals():
        max_distance = df["distance_km"].max()
        slice_distance = st.number_input(
            "How many km would you like the lap to be?",
            min_value=1,
            max_value=int(math.ceil(max_distance)),
            value=10
        )

        if st.button("Generate Laps"):
            laps = []
            start_dist = 0.0
            lap_num = 1
            total_distance = round(max_distance, 3)  # round to avoid floating point errors

            while start_dist < total_distance:
                # For last lap, use remaining distance
                end_dist = start_dist + slice_distance
                if end_dist > total_distance:
                    end_dist = total_distance

                # Round distances for comparison to avoid floating point issues
                lap_df = df[(df["distance_km"].round(3) >= round(start_dist, 3)) &
                            (df["distance_km"].round(3) <= round(end_dist, 3))].copy()
                if lap_df.empty:
                    break

                # Calculate start/end time in HH:MM
                start_sec = lap_df["elapsed_sec"].iloc[0]
                end_sec = lap_df["elapsed_sec"].iloc[-1]
                duration_sec = end_sec - start_sec
                start_hhmm = seconds_to_hhmm(start_sec)
                end_hhmm = seconds_to_hhmm(end_sec)
                duration_hhmm = seconds_to_hhmm(duration_sec)

                # Correct lap distance (last lap may be shorter)
                lap_distance = round(end_dist - start_dist, 3)

                laps.append({
                    "name": f"Lap {lap_num}",
                    "start_time": start_hhmm,
                    "end_time": end_hhmm,
                    "duration": duration_hhmm,
                    "start_idx": lap_df.index[0],
                    "end_idx": lap_df.index[-1],
                    "distance": lap_distance
                })

                start_dist += slice_distance
                lap_num += 1

            # Store laps in session_state for further analysis
            st.session_state["lap_data"] = laps
            st.session_state["lap_form_submitted"] = True

            st.subheader("Distance Slicer Laps")
            if laps:
                lap_df_display = pd.DataFrame(laps)[["name", "start_time", "end_time", "duration", "distance"]]
                st.dataframe(lap_df_display)
            else:
                st.warning("No laps generated. Check the distance slice or your FIT file.")
    else:
        st.info("👆 Please upload a .fit file first to use the Distance Slicer.")
# ---------------------------
# MANUAL LAP INPUT
# ---------------------------
if (st.session_state.get("do_lap_analysis") and st.session_state.get("climb_data_insert") in ["manual"]) \
   or st.session_state.get("lap_data_insert") in ["manual"]:

    num_laps = st.number_input(
        f"How many {st.session_state.get('analysis_type','')} would you like to analyze?",
        min_value=1,
        max_value=20,
        step=1,
        key="num_laps"
    )

    # Dynamic form
    with st.form("laps_form"):
        st.subheader(f"📋 {st.session_state.get('analysis_type','').capitalize()} Details")
        lap_inputs = []

        for i in range(num_laps):
            st.markdown(f"### {st.session_state.get('analysis_type','').capitalize()} {i+1}")
            lap_name = st.text_input(
                f"{st.session_state.get('analysis_type','').capitalize()} {i+1} name",
                value=f"{st.session_state.get('analysis_type','').capitalize()} {i+1}",
                key=f"lap_name_{i}"
            )
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.text_input(
                    f"{st.session_state.get('analysis_type','').capitalize()} {i+1} Start Time [HH:MM]",
                    value="",
                    key=f"start_time_{i}",
                    placeholder="es. 00:30"
                )
            with col2:
                end_time = st.text_input(
                    f"{st.session_state.get('analysis_type','').capitalize()} {i+1} End Time [HH:MM]",
                    value="",
                    key=f"end_time_{i}",
                    placeholder="es. 02:00"
                )
            ngp = st.text_input(
                f"{st.session_state.get('analysis_type','').capitalize()} {i+1} NGP [mm:ss] (optional)",
                value="",
                key=f"ngp_{i}",
                placeholder="es. 04:30"
            )

            lap_inputs.append({
                "name": lap_name,
                "start_time": start_time,
                "end_time": end_time,
                "ngp": ngp
            })

        submitted = st.form_submit_button(f"Submit {st.session_state.get('analysis_type','').capitalize()} Data")
        if submitted:
            st.session_state["lap_form_submitted"] = True
            st.session_state["lap_data"] = lap_inputs
            st.success(f"✅ {st.session_state.get('analysis_type','').capitalize()} data submitted successfully!")

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
        - 🩵 Zone 1 (Recovery): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Tempo): {z2+1} - {z3} bpm  
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
                return "Zone 1 // Recovery"
            elif hr <= z2:
                return "Zone 2 // Aerobic"
            elif hr <= z3:
                return "Zone 3 // Tempo"
            elif hr <= z4:
                return "Zone 4 // Sub Threshold"
            else:
                return "Zone 5 // Super Threshold"

        df["HR Zone"] = df["heart_rate"].apply(get_hr_zone)
        df["time_diff_sec"] = df["elapsed_sec"].diff().fillna(0)

        zone_order = ["Zone 1 // Recovery","Zone 2 // Aerobic","Zone 3 // Tempo","Zone 4 // Sub Threshold","Zone 5 // Super Threshold"]

        # Total (overall) time-in-zone
        total_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)

        # Time segment warning

        segment_keys = [
            'segment1_start','segment1_end',
            'segment2_start','segment2_end',
            'segment3_start','segment3_end'
        ]

        if not all(k in st.session_state for k in segment_keys):
            missing_seg = [k for k in segment_keys if k not in st.session_state]
            st.warning(f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis.")

        # Prepare segment inputs with formatted names
        segment_inputs = []    
        for i in range(1, 4):
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

        # Format as H:MM
        combined_df = combined_df.applymap(lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}")

        st.markdown("### ⏱️ Time-in-Zone Analysis")
        st.dataframe(combined_df)

    else:
        st.warning("⚠️ Please submit the Heart Rate Zones to enable Time-in-Zone analysis.")    

# -----------------------------
# LAP / CLIMB ANALYSIS
# -----------------------------
if 'df' in locals() and not df.empty:
    if st.session_state.get("do_lap_analysis") and 'lap_data' in st.session_state:
        lap_data = st.session_state.get("lap_data", [])

        if 'HR Zone' in df.columns:
            # Map full HR zone names to short names
            hr_zone_map = {
                "Zone 1 // Recovery": "Z1",
                "Zone 2 // Aerobic": "Z2",
                "Zone 3 // Tempo": "Z3",
                "Zone 4 // Sub Threshold": "Z4",
                "Zone 5 // Super Threshold": "Z5"
            }

            zone_order = ["Z1", "Z2", "Z3", "Z4", "Z5"]
            lap_zone_data = []

            def h_mm_to_seconds(hmm):
                """Convert H:MM to seconds."""
                try:
                    h, m = map(int, hmm.split(":"))
                    return h*3600 + m*60
                except:
                    return 0

            for lap in lap_data:
                # --- Compute start/end seconds ---
                start_sec = h_mm_to_seconds(lap.get("start_time", "0:00"))
                end_sec = h_mm_to_seconds(lap.get("end_time", "0:01"))
                duration_sec = max(end_sec - start_sec, 1)
                duration_hm = f"{int(duration_sec//3600)}:{int((duration_sec%3600)//60):02d}"

                # --- Slice dataframe for this lap ---
                if "start_idx" in lap and "end_idx" in lap:
                    df_lap = df.loc[lap["start_idx"]:lap["end_idx"]].copy()
                else:
                    df_lap = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()

                df_lap["time_diff_sec"] = df_lap["elapsed_sec"].diff().fillna(0)
                df_lap["HR Zone Short"] = df_lap["HR Zone"].map(hr_zone_map)

                # --- HR zone summary ---
                lap_summary = df_lap.groupby("HR Zone Short")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
                lap_summary_hm = [f"{int(x//3600)}:{int((x%3600)//60):02d}" for x in lap_summary.values]
                pct_zones = [f"{round((x/duration_sec)*100)}%" for x in lap_summary.values]

                # --- Average HR ---
                avg_fc = int(df_lap["heart_rate"].mean()) if not df_lap.empty else 0

                # --- Distance and elevation gain dynamically ---
                distance = round(df_lap["distance_km"].max() - df_lap["distance_km"].min(),1) if "distance_km" in df_lap else 0
                elevation = df_lap["elevation_m"].diff().clip(lower=0).sum() if "elevation_m" in df_lap else 0

                # --- NGP ---
                ngp = lap.get("ngp", "")

                # --- Build row depending on analysis type ---
                if st.session_state.get("analysis_type") == "climb":
                    avg_grade = round((elevation / distance / 10) if distance > 0 else 0)
                    vam = round(elevation / (duration_sec / 3600) if duration_sec > 0 else 0)
                    lap_zone_data.append([
                        lap.get("name", "Lap"), duration_hm, distance, int(elevation),
                        avg_fc, avg_grade, vam, ngp
                    ] + lap_summary_hm + pct_zones)
                else:  # Lap analysis
                    pace_min_float = duration_sec / 60 / distance if distance > 0 else 0
                    pace_min = int(pace_min_float)
                    pace_sec = int(round((pace_min_float - pace_min) * 60))
                    lap_pace = f"{pace_min:02d}:{pace_sec:02d}" if distance > 0 else "00:00"

                    lap_zone_data.append([
                        lap.get("name", "Lap"), duration_hm, distance, int(elevation),
                        avg_fc, lap_pace, ngp
                    ] + lap_summary_hm + pct_zones)

            # --- Build DataFrame ---
            if st.session_state.get("analysis_type") == "climb":
                extra_cols = ["Avg FC", "Avg Grade (%)", "VAM (m/h)", "NGP"]
            else:
                extra_cols = ["Avg FC", "Lap Pace (min/km)", "NGP"]

            pct_cols = [f"% {z}" for z in zone_order]
            columns = [f"{st.session_state.get('analysis_type','').replace(' Analysis','').capitalize()} Name",
                       "Duration", "Distance (km)", "Elevation (m)"] + extra_cols + zone_order + pct_cols

            lap_zone_df = pd.DataFrame(lap_zone_data, columns=columns)
            if "Distance (km)" in lap_zone_df.columns:
                lap_zone_df["Distance (km)"] = lap_zone_df["Distance (km)"].astype(float).map("{:.1f}".format)
            # Build DataFrame
            lap_zone_df = pd.DataFrame(lap_zone_data, columns=columns)
            if "Distance (km)" in lap_zone_df.columns:
                lap_zone_df["Distance (km)"] = lap_zone_df["Distance (km)"].astype(float).map("{:.1f}".format)

            # Make editable
            st.markdown(f"### {analysis_type}")
            st.markdown("*You can edit the table clicking the data you want to change*")
            edited_df = st.data_editor(
                lap_zone_df,
                num_rows="dynamic",
                use_container_width=True
                )

            # Store edits for later use
            st.session_state["lap_zone_data_edited"] = edited_df


        else:
            st.warning("⚠️ Please submit the Heart Rate Zones first to enable Lap/Climb analysis.")
    else:
        st.info("👆 No lap/climb data to analyze yet. Fill the form above.")
else:
    st.warning("⚠️ Please upload a FIT file first to perform analysis.")

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

    # HEATMAP GRAPH IN MINUTES
    # Convert H:MM to minutes
    def h_mm_to_minutes(hmm_str):
        try:
            h, m = map(int, hmm_str.split(":"))
            return h*60 + m
        except:
            return 0

    heatmap_df_minutes = bar_df.applymap(h_mm_to_minutes)
    heatmap_df_minutes.index = [f"Zone {i+1}" for i in range(len(heatmap_df_minutes))]
    heatmap_df_minutes_reset = heatmap_df_minutes.reset_index().rename(columns={'index':'HR Zone'})
    heatmap_long = heatmap_df_minutes_reset.melt(id_vars="HR Zone", var_name="Segment", value_name="Minutes")

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
    fig_heat.update_layout(
        yaxis=dict(autorange='reversed'),
        coloraxis_colorbar=dict(title="Time (minutes)")
    )

    st.plotly_chart(fig_heat, use_container_width=True)

else:
    st.warning(" ⚠️ Please submit all data required for the analysis.")

# --- Plotly chart for % time in HR zones per Lap/Climb ---
if st.session_state.get("do_lap_analysis") and 'lap_zone_df' in locals():

    # Define % columns for HR zones
    hr_zones = ["Z1","Z2","Z3","Z4","Z5"]
    pct_cols = [f"% {z}" for z in hr_zones]

    # Determine the correct name column in lap_zone_df
    if "Lap Name" in lap_zone_df.columns:
        name_col = "Lap Name"
    elif "Climb Name" in lap_zone_df.columns:
        name_col = "Climb Name"
    else:
        st.warning("Could not find the Name column in lap_zone_df.")
        name_col = None

    if name_col:
        # Select only name and % columns
        df_plot = lap_zone_df[[name_col] + pct_cols].copy()

        # Convert percentage strings to numeric
        for col in pct_cols:
            df_plot[col] = df_plot[col].str.rstrip('%').astype(float)

        # Melt the DataFrame for Plotly
        df_melted = df_plot.melt(
            id_vars=[name_col],
            value_vars=pct_cols,
            var_name="HR Zone",
            value_name="Percentage"
        )

        # Plot
        fig = px.bar(
            df_melted,
            x="HR Zone",
            y="Percentage",
            color=name_col,
            barmode="group",
            title=f"{analysis_type} - % Time in HR Zones"
        )

        st.plotly_chart(fig)

else:
    st.info("👆 No lap/climb data to analyze yet. Fill the form above.")

import streamlit as st

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
        self.set_xy(10, 18)  # move cursor below main title
        self.set_font("Helvetica", "I", 11)  # italic or lighter font
        self.set_text_color(200, 200, 200)   # lighter gray
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

# --------------------------------------------------------------------
# PDF GENERATION CLASS
# --------------------------------------------------------------------
def add_chart_to_pdf(fig, title=None):
    if title:
        pdf.section_title(title)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="PNG", dpi=200, bbox_inches="tight")
        tmpfile.close()
        pdf.image(tmpfile.name, x=10, w=190)
    plt.close(fig)


class ModernPDF(FPDF):
    """Custom PDF class for DU Coaching Race Analyzer."""
    
    def header(self):
        # Header bar
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 30, 'F')
        self.set_xy(10, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 10, "DU COACHING - Race Analyzer Report", ln=True)
        
        # Subtitle
        self.set_xy(10, 18)
        self.set_font("Helvetica", "I", 11)
        self.set_text_color(200, 200, 200)
        self.cell(0, 6, "This analyzer is brought to you by Coach Ambro", ln=True)
        self.ln(5)
    
    def section_title(self, title: str):
        """Add a section title with background fill."""
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 30)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 10, title, ln=True, fill=True)
        self.ln(4)
    
    def body_text(self, text: str):
        """Add a paragraph of body text."""
        self.set_font("Helvetica", "", 11)
        self.set_text_color(55, 55, 55)
        self.multi_cell(0, 6, text)
        self.ln(2)
    
    def add_spacer(self, height: int = 4):
        """Add vertical space."""
        self.ln(height)


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
        pdf.add_spacer(6)

        # --- Time-in-Zone Table ---
        if "combined_df" in locals():
            pdf.section_title("Time-in-Zone Table [hh:mm]")
            page_w = pdf.w - pdf.l_margin - pdf.r_margin
            n_cols = 1 + len(combined_df.columns)
            col_width = page_w / n_cols
            row_height = 8

            # Header
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(245, 245, 245)
            pdf.set_text_color(20, 20, 20)
            pdf.cell(col_width, row_height, "HR Zone", border=1, fill=True)
            for col in combined_df.columns:
                pdf.cell(col_width, row_height, str(col), border=1, fill=True)
            pdf.ln()

            # Rows
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(60, 60, 60)
            for i, zone in enumerate(combined_df.index, 1):
                pdf.cell(col_width, row_height, f"Zone {i}", border=1)
                for seg in combined_df.columns:
                    pdf.cell(col_width, row_height, str(combined_df.loc[zone, seg]), border=1)
                pdf.ln()

            pdf.add_spacer(3)

        # --- Add Bar Chart & Heatmap ---
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
            ax.legend(title="Segment")  # Added legend with title
            plt.tight_layout()
            add_chart_to_pdf(fig, title="Time-in-Zone - Bar Chart")


        if "heatmap_df_minutes" in locals():
            fig, ax = plt.subplots(figsize=(10, 3 + max(0, len(heatmap_df_minutes)/4)))
            sns.heatmap(
                heatmap_df_minutes,
                annot=True,
                fmt="d",
                cmap="YlOrRd",
                linewidths=0.5,
                linecolor='white',
                cbar_kws={'label':'Minutes'},
                ax=ax
            )
            ax.set_title("Time-in-Zone Heatmap (Minutes)")
            ax.set_ylabel("HR Zone")
            ax.set_xlabel("Segment")
            plt.tight_layout()
            add_chart_to_pdf(fig, title="Time-in-Zone - Heatmap")
        pdf.add_page()

        # --- Elevation Profile with Climbs ---

        if 'df' in locals() and not df.empty and "climb_data" in st.session_state:
            climbs = st.session_state["climb_data"]

            fig, ax = plt.subplots(figsize=(10, 4))

            # Plot full elevation in gray
            ax.plot(df["elapsed_sec"], df["elevation_m"], color="gray", label="Elevation")

            # Overlay climbs in green
            for c in climbs:
                s = c["start_idx"]
                e = c["end_idx"]
                ax.plot(df["elapsed_sec"].iloc[s:e+1], df["elevation_m"].iloc[s:e+1],
                        color="green", linewidth=2, label=c.get("name", "Climb"))

            ax.set_xlabel("Elapsed Time (sec)")
            ax.set_ylabel("Elevation (m)")
            ax.set_title("Elevation Profile with Climbs Highlighted")
            ax.grid(True)

            # Avoid duplicate legend entries
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys())

            plt.tight_layout()

            # Insert into PDF
            add_chart_to_pdf(fig, title="Elevation Profile with Climbs")

        # --- Lap / Climb Table ---
        if st.session_state.get("do_lap_analysis") and "lap_zone_df" in locals():
            pdf.section_title(f"{analysis_type} - Info Table")

            lap_zone_df_copy = lap_zone_df.copy()
            info_cols = [col for col in lap_zone_df_copy.columns if not col.startswith("Z") and not col.startswith("%")]
            n_cols = len(info_cols)
            col_width = (pdf.w - pdf.l_margin - pdf.r_margin) / n_cols
            row_height = 6

            # Header for Info Table
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(245, 245, 245)
            for col in info_cols:
                pdf.cell(col_width, row_height, str(col)[:12], border=1, fill=True, align='C')
            pdf.ln(row_height)

            # Rows for Info Table
            pdf.set_font("Helvetica", "", 10)
            for _, row in lap_zone_df_copy.iterrows():
                for col in info_cols:
                    pdf.cell(col_width, row_height, str(row[col]), border=1, align='C')
                pdf.ln(row_height)
            pdf.add_spacer(6)

            # --- HR Data Analysis Table ---
            pdf.section_title(f"{analysis_type} - HR Data Analysis")

            hr_cols = [f"{analysis_type[:-8]} name"] + [f"Z{i}" for i in range(1,6)] + [f"% Z{i}" for i in range(1,6)]
            n_cols = len(hr_cols)
            col_width = (pdf.w - pdf.l_margin - pdf.r_margin) / n_cols
            row_height = 6

            # Header
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_fill_color(245, 245, 245)
            for col in hr_cols:
                pdf.cell(col_width, row_height, col, border=1, fill=True, align='C')
            pdf.ln(row_height)

            # Rows
            pdf.set_font("Helvetica", "", 10)
            for _, row in lap_zone_df_copy.iterrows():
                # Use same lap name as in first table
                pdf.cell(col_width, row_height, str(row[info_cols[0]]), border=1, align='C')  # lap name
                for i in range(1, 6):
                    pdf.cell(col_width, row_height, str(row.get(f"Z{i}", "")), border=1, align='C')
                for i in range(1, 6):
                    pdf.cell(col_width, row_height, str(row.get(f"% Z{i}", "")), border=1, align='C')
                pdf.ln(row_height)

            pdf.add_spacer(6)

                    # --- Lap/Climb % Time-in-Zone Chart ---
        if st.session_state.get("do_lap_analysis") and 'lap_zone_df' in locals() and not lap_zone_df.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            
            zones = ["Z1", "Z2", "Z3", "Z4", "Z5"]
            n_laps = len(lap_zone_df)
            x = np.arange(len(zones))
            bar_width = 0.8 / max(n_laps, 1)  # prevent division by zero
            
            for i, (_, row) in enumerate(lap_zone_df.iterrows()):
                pct_values = []
                for z in zones:
                    val = row.get(f"% {z}", "0")
                    if isinstance(val, str):
                        val = val.rstrip('%')
                    pct_values.append(float(val or 0))
                ax.bar(x + i * bar_width, pct_values, width=bar_width, label=row.get("name", f"Lap {i+1}"))
            
            # Center x-ticks
            ax.set_xticks(x + bar_width * (n_laps - 1) / 2)
            ax.set_xticklabels(zones)
            
            ax.set_ylabel("Percentage (%)")
            ax.set_title(f"{analysis_type.capitalize()} - % Time in HR Zones")
            ax.legend()
            
            plt.tight_layout()
            add_chart_to_pdf(fig, title=f"{analysis_type.capitalize()} - % Time in HR Zones")

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

# -------- COACH COMMENT SECTION --------

        if st.session_state.comment:
            pdf.section_title("Coach comment")
            pdf.body_text(f"{st.session_state.comment}")

        pdf_data = pdf.output(dest="S")
        if isinstance(pdf_data, str):
            pdf_bytes = pdf_data.encode('latin1')  # FPDF string -> bytes
        elif isinstance(pdf_data, (bytes, bytearray)):
            pdf_bytes = bytes(pdf_data)           # already bytes-like
        else:
            raise TypeError(f"Unexpected type from FPDF output: {type(pdf_data)}")

        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state['athlete_name']}_{st.session_state['race_name']}_{st.session_state['race_date'].year}_Race_report.pdf",
            mime="application/pdf"
        )
else:
    st.warning ("⚠️ Please submit race, athlete and cardiac data to generate the PDF report")