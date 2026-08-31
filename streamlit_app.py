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
import warnings
import fitdecode
import gzip
import os
import smtplib
from email.message import EmailMessage
import re

LOGO_DU_PATH   = "LOGO_DU_BIANCO.jpeg"
SCRITTA_PATH   = "SCRITTA_ULTRANERD.png"

st.set_page_config(page_title="DU COACHING RACE Analyzer", layout="wide")

# --- Intestazione: i due loghi affiancati e centrati, poi il titolo ---
# Le immagini sono inlined in base64 dentro un unico blocco HTML flex:
# st.columns + st.image non permette di centrare davvero (ogni immagine si
# centra nella propria colonna) e lascia margini verticali non controllabili.
import base64

LOGO_DU_HEIGHT_PX = 180    # stessa altezza per entrambi -> allineamento ottico
SCRITTA_HEIGHT_PX = 164    # la scritta è più "leggera", va un filo più bassa

def _img_tag(path: str, height_px: int) -> str:
    if not os.path.exists(path):
        return ""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f"<img src='data:{mime};base64,{b64}' "
            f"style='height:{height_px}px; width:auto; display:block;'/>")

st.markdown(
    "<div style='display:flex; align-items:center; justify-content:center; "
    "gap:28px; margin:0 0 6px 0;'>"
    + _img_tag(LOGO_DU_PATH, LOGO_DU_HEIGHT_PX)
    + _img_tag(SCRITTA_PATH, SCRITTA_HEIGHT_PX)
    + "</div>"
    "<h1 style='text-align:center; margin:0 0 0.4em 0; padding-top:0;'>"
    "📊 DU COACHING RACE Analyzer 📊</h1>",
    unsafe_allow_html=True
)

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
# ---------------------------------------------------------------------------
# EFS (Equivalent Flat Speed) — energy cost of running, Minetti et al. (2002)
# ---------------------------------------------------------------------------
EFS_RESAMPLE_STEP_M = 20.0   # passo della griglia orizzontale
EFS_SMOOTH_WINDOW   = 9      # punti di rolling mean sulla quota
EFS_MIN_KMH = 0.5            # sotto = soste/ristori, non velocità reale
EFS_MAX_KMH = 30.0           # sopra = glitch GPS/quota
EFS_PLOT_SMOOTH_PTS = 61   # punti di rolling median per la linea a schermo
EFS_AXIS_HEADROOM   = 2.4  # >1 spinge la curva EFS in basso, lontano dalla FC
ELE_AXIS_HEADROOM = 4.5   # >1 schiaccia il profilo in basso, sotto le curve
SPD_INDEX_SCALE = 1e4      # costante di scala dell'indice SPD (vedi commento nel calcolo)

def cost_of_running(slope: np.ndarray) -> np.ndarray:
    """Costo energetico specifico della corsa, J/(kg*m), in funzione della
    pendenza (frazione decimale). Vettoriale, clampato a +/-45%."""
    i = np.clip(slope, -0.45, 0.45)
    return (155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3
            + 46.3 * i**2 + 19.5 * i + 3.6)

FLAT_COST = float(cost_of_running(np.array([0.0]))[0])  # 3.6 J/kg/m


def compute_efs_series(df, resample_step_m=EFS_RESAMPLE_STEP_M,
                       smooth_window=EFS_SMOOTH_WINDOW):
    """
    Calcola l'EFS segmento per segmento usando la distanza del dispositivo
    (df['distance_km']) come asse orizzontale, quindi senza ricalcolare le
    distanze da lat/lon.

    Ritorna (efs_df, totals):
      efs_df  -> elapsed_hours (centro segmento), efs_kmh, slope_pct, km
      totals  -> dict con efd_km, avg_efs_kmh, total_time_s
    """
    if df is None or df.empty or "distance_km" not in df.columns:
        return None, None

    # la distanza del dispositivo può avere micro-arretramenti: si forza
    # monotona, altrimenti np.interp riceve un xp non crescente
    cum_dist = np.maximum.accumulate(
        pd.to_numeric(df["distance_km"], errors="coerce").ffill().fillna(0).to_numpy() * 1000.0
    )
    time_s = pd.to_numeric(df["elapsed_sec"], errors="coerce").ffill().fillna(0).to_numpy()
    ele = (pd.to_numeric(df["elevation_m"], errors="coerce")
             .ffill().bfill().fillna(0.0)
             .rolling(window=smooth_window, center=True, min_periods=1)
             .mean().to_numpy())

    total_dist = float(cum_dist[-1])
    if total_dist <= 0 or len(cum_dist) < 2:
        return None, None

    n_steps = max(2, int(total_dist // resample_step_m))
    grid = np.linspace(0.0, total_dist, n_steps)
    ele_grid = np.interp(grid, cum_dist, ele)
    time_grid = np.interp(grid, cum_dist, time_s)

    dx = np.diff(grid)
    dz = np.diff(ele_grid)
    dt = np.diff(time_grid)
    dist3d = np.hypot(dx, dz)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)

    efd_m = cost_of_running(slope) * dist3d / FLAT_COST
    efs_ms = np.divide(efd_m, dt, out=np.full_like(efd_m, np.nan), where=dt > 0)
    efs_kmh = efs_ms * 3.6

    # soste e glitch fuori: restano NaN, così non spezzano la serie né
    # sporcano la regressione
    efs_kmh = np.where((efs_kmh >= EFS_MIN_KMH) & (efs_kmh <= EFS_MAX_KMH),
                       efs_kmh, np.nan)

    mid_time_h = (time_grid[:-1] + time_grid[1:]) / 2.0 / 3600.0

    efs_df = pd.DataFrame({
        "km": grid[1:] / 1000.0,
        "elapsed_hours": mid_time_h,
        "efs_kmh": efs_kmh,
        "slope_pct": slope * 100.0,
    })

    total_time_s = float(np.nansum(dt))
    totals = {
        "efd_km": float(efd_m.sum() / 1000.0),
        "total_time_s": total_time_s,
        "avg_efs_kmh": (efd_m.sum() / total_time_s * 3.6) if total_time_s > 0 else np.nan,
    }
    return efs_df, totals

def compute_segment_efs(df_seg, resample_step_m=EFS_RESAMPLE_STEP_M,
                        smooth_window=EFS_SMOOTH_WINDOW):
    """EFS media di una porzione di traccia (km/h), pesata sul tempo:
    EFD totale del tratto / tempo totale del tratto. Non è la media delle
    EFS istantanee, che sovrapeserebbe i segmenti lenti.
    Ritorna np.nan se il tratto è troppo corto o degenere."""
    if df_seg is None or len(df_seg) < 3 or "distance_km" not in df_seg.columns:
        return np.nan
    d = np.maximum.accumulate(
        pd.to_numeric(df_seg["distance_km"], errors="coerce").ffill().fillna(0).to_numpy() * 1000.0
    )
    d = d - d[0]
    t = pd.to_numeric(df_seg["elapsed_sec"], errors="coerce").ffill().fillna(0).to_numpy()
    e = (pd.to_numeric(df_seg["elevation_m"], errors="coerce")
           .ffill().bfill().fillna(0.0)
           .rolling(window=smooth_window, center=True, min_periods=1)
           .mean().to_numpy())

    total = float(d[-1])
    if total <= 0:
        return np.nan

    n = max(2, int(total // resample_step_m))
    grid = np.linspace(0.0, total, n)
    ele_g = np.interp(grid, d, e)
    t_g = np.interp(grid, d, t)

    dx, dz, dt = np.diff(grid), np.diff(ele_g), np.diff(t_g)
    slope = np.divide(dz, dx, out=np.zeros_like(dz), where=dx > 0)
    efd_m = cost_of_running(slope) * np.hypot(dx, dz) / FLAT_COST

    total_t = float(dt.sum())
    if total_t <= 0:
        return np.nan
    return float(efd_m.sum() / total_t * 3.6)

# Soglie empiriche sul delta EF (2ª metà - 1ª metà): p25 = -1.76,
# p75 = -0.91. Sopra il p75 la perdita è bassa, sotto il p25 è
# significativa. Il delta dice più del livello assoluto di EF, che
# dipende dalla gara (dislivello, superficie, meteo).
EF_DELTA_P25 = -1.76
EF_DELTA_P75 = -0.91


def ef_delta_verdict(delta):
    """(commento, colore di sfondo) per il calo di EF fra le due metà."""
    if delta >= EF_DELTA_P75:
        return "Bassa perdita di efficienza", "#7ddf9a"      # verde
    if delta >= EF_DELTA_P25:
        return "Perdita di efficienza nella norma", "#f0f0f0"  # bianco
    return "Perdita di efficienza significativa", "#f2d14b"   # giallo


def ef_metric_box(label, value, bg="rgba(128,128,128,0.14)", fg="inherit"):
    """Riquadro etichetta+valore per la riga dei dati EF."""
    return (f"<div style='background-color:{bg}; color:{fg}; padding:8px 16px; "
            f"border-radius:6px; text-align:center; min-width:118px;'>"
            f"<div style='font-size:12px; opacity:0.75;'>{label}</div>"
            f"<div style='font-size:22px; font-weight:700; line-height:1.2;'>"
            f"{value}</div></div>")

st.write("")
st.markdown("*For large race files, to speed up the analysis, first add the race and cardiac data, and then upload the .fit or .gzip file*")
uploaded_file = st.file_uploader("Upload a .fit or .fit.gz file", type=["fit", "gz"])


@st.cache_data(show_spinner="⏳ Parsing FIT file (this only happens once per file)...")
def parse_fit_file(file_bytes):
    """
    Heavy one-time parsing of the raw .fit bytes into a clean DataFrame.
    Uses fitdecode instead of fitparse: some devices (e.g. COROS) write
    malformed field-size declarations in definition messages. fitparse
    treats this as fatal and aborts the whole file; fitdecode logs a
    warning and skips just that field, so the rest of the file still
    parses normally.
    """
    # --- transparent gzip support ---
    if file_bytes[:2] == b"\x1f\x8b":
        try:
            file_bytes = gzip.decompress(file_bytes)
        except Exception as e:
            raise ValueError(f"Could not decompress gzip file: {e}")

    records = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        with fitdecode.FitReader(io.BytesIO(file_bytes)) as fit:
            for frame in fit:
                if frame.frame_type != fitdecode.FIT_FRAME_DATA or frame.name != "record":
                    continue
                row = {f.name: f.value for f in frame.fields
                       if f.name in ["timestamp", "heart_rate", "distance",
                                      "enhanced_altitude", "position_lat", "position_long"]}
                if row:
                    records.append(row)

    if not records:
        raise ValueError("No usable records in this FIT file.")

    df = pd.DataFrame(records)

    # Fill missing columns
    for col in ["heart_rate", "distance", "enhanced_altitude", "position_lat", "position_long"]:
        df[col] = df.get(col, np.nan)

    # Convert units
    df["distance_km"] = df["distance"].apply(lambda x: x/1000 if pd.notna(x) else np.nan)
    df["distance_km"] = df["distance_km"].ffill().fillna(0).astype(float)

    df["elevation_m"] = df["enhanced_altitude"].ffill().fillna(0).astype(float)

    df["lat"] = df["position_lat"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)
    df["lon"] = df["position_long"].apply(lambda s: s*(180/2**31) if pd.notna(s) else np.nan)

    if "timestamp" in df.columns and not df["timestamp"].isna().all():
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        start_time = df["timestamp"].iloc[0]
        df["elapsed_sec"] = (df["timestamp"] - start_time).dt.total_seconds()
    else:
        df["elapsed_sec"] = np.arange(len(df))

    df["time_diff_sec"] = df["elapsed_sec"].diff().clip(lower=0).fillna(0)
    df["elapsed_hours"] = df["elapsed_sec"] / 3600
    df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean() if "heart_rate" in df.columns else np.nan

    kilometers = df["distance_km"].max() if "distance_km" in df.columns else 0
    total_elevation_gain = df["elevation_m"].diff().clip(lower=0).sum() if "elevation_m" in df.columns else 0

    return df, int(kilometers), int(total_elevation_gain)

if uploaded_file is not None:
    st.success("FIT file uploaded!")
    try:
        file_bytes = uploaded_file.getvalue()
        df, kilometers, total_elevation_gain = parse_fit_file(file_bytes)

        # Save results in session state so the rest of the app can use them
        # without needing to touch the (cached) parsing function again
        st.session_state['fit_df'] = df
        st.session_state['kilometers'] = kilometers
        st.session_state['total_elevation_gain'] = total_elevation_gain

    except Exception as e:
        st.error(f"❌ Error reading FIT file: {e}")
else:
    st.info("👆 Please upload a .fit file to begin.")

hr_zones_ready = bool(st.session_state.get("hr_zones_confirmed"))
analysis_ready = (uploaded_file is not None) and hr_zones_ready

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
        # I default 140/160/170/180/200 sono già in session_state all'avvio:
        # senza questo flag non c'è modo di distinguere zone confermate
        # dall'atleta da valori mai toccati.
        st.session_state['hr_zones_confirmed'] = True
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
# ---------------------------------------------------------------------------
# GATE dell'analisi
# ---------------------------------------------------------------------------
# I default 140/160/170/180/200 sono in session_state fin dall'avvio: senza
# questo flag l'app produrrebbe tempi-in-zona, densità e EF perfettamente
# plausibili ma calcolati su valori che l'atleta non ha mai confermato.
# Il flag lo alza SOLO il bottone "Submit HR Zones".
hr_zones_ready = bool(st.session_state.get("hr_zones_confirmed"))
analysis_ready = (uploaded_file is not None) and hr_zones_ready

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
    
def h_mm_to_seconds(hmm):
    """'1:30' -> 5400. None se non parsabile."""
    try:
        h, m = str(hmm).split(":")
        return int(h) * 3600 + int(m) * 60
    except (ValueError, AttributeError):
        return None


def format_hmm(hmm):
    """'0:00' -> '0', '1:30' -> \"1h30'\". Etichette dei segmenti."""
    try:
        h, m = str(hmm).split(":")
        h, m = int(h), int(m)
        return "0" if (h == 0 and m == 0) else f"{h}h{m:02d}'"
    except (ValueError, AttributeError):
        return hmm

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

# ---------------------------------------------------------------------------
# Invio file grezzi (FIT + zone FC) via email
# ---------------------------------------------------------------------------

def build_archive_payload(fit_bytes, fit_filename, anonymous=False):
    """Ritorna [(filename, bytes, mimetype), ...] con il FIT originale e il
    CSV delle zone FC rigenerato da session_state. Il CSV si ricostruisce
    invece di riusare l'upload perché in modalità Manual Input il file non
    esiste, e comunque le zone possono essere state modificate a mano."""
    ext = ".fit.gz" if fit_filename.lower().endswith(".gz") else os.path.splitext(fit_filename)[1]

    if anonymous:
        # Niente nome, gara o data: né nei filename né dentro il CSV.
        # La data di gara da sola, incrociata con i risultati pubblici,
        # basterebbe a risalire all'atleta.
        out = [(f"race_data{ext}", fit_bytes, "application/octet-stream")]
        zone_row = {}
    else:
        athlete = (st.session_state.get("athlete_name") or "athlete").strip().replace(" ", "_")
        race    = (st.session_state.get("race_name") or "race").strip().replace(" ", "_")
        stamp   = st.session_state.get("race_date")
        stamp   = stamp.strftime("%Y%m%d") if stamp else "nodate"
        prefix  = f"{athlete}_{race}_{stamp}"
        out = [(f"{prefix}{ext}", fit_bytes, "application/octet-stream")]
        zone_row = {"athlete_name": st.session_state.get("athlete_name", "")}

    if all(k in st.session_state for k in ("z1", "z2", "z3", "z4", "z5")):
        zone_row.update({
            "z1": st.session_state["z1"], "z2": st.session_state["z2"],
            "z3": st.session_state["z3"], "z4": st.session_state["z4"],
            "z5": st.session_state["z5"],
        })
        zones_csv = pd.DataFrame([zone_row]).to_csv(index=False).encode("utf-8")
        csv_name = "hr_zones.csv" if anonymous else f"{prefix}_HR_Zones.csv"
        out.append((csv_name, zones_csv, "text/csv"))

    return out

def missing_archive_requirements():
    """Lista dei dati mancanti per l'invio. Vuota = si può inviare.
    Il file .fit non è qui dentro: quel controllo lo fa già il blocco
    chiamante, che senza upload non disegna nemmeno il bottone."""
    missing = []
    if not str(st.session_state.get("athlete_name", "")).strip():
        missing.append("Athlete's Name")
    if not str(st.session_state.get("race_name", "")).strip():
        missing.append("Race name")
    if st.session_state.get("race_date") is None:
        missing.append("Race date")
    if not st.session_state.get("hr_zones_confirmed"):
        missing.append("HR Zones (press 'Submit HR Zones')")
    return missing

def missing_anonymous_requirements():
    """Per l'invio anonimo servono solo le zone FC: nome atleta, gara e
    data non finiscono da nessuna parte, quindi non ha senso pretenderli."""
    if not st.session_state.get("hr_zones_confirmed"):
        return ["HR Zones (press 'Submit HR Zones')"]
    return []


def send_payload_email(payload, subject, body):
    """SMTP con STARTTLS. L'eccezione risale al chiamante: l'errore va
    mostrato all'utente con st.error, non ingoiato qui dentro."""
    cfg = st.secrets["email"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(body)

    for fname, data, mime in payload:
        maintype, _, subtype = mime.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream",
                           filename=fname)

    with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=60) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)
    return len(payload)

# ---------------------------------------------------------------------------
# Climb detection (v2)
# ---------------------------------------------------------------------------
# Il rilevatore precedente lavorava sulla quota GREZZA e chiudeva la salita
# quando la discesa cumulata superava una soglia fissa. Con il rumore
# barometrico (±1-2 m per campione) questo significa: guadagno gonfiato dal
# rumore, salite che si chiudono a caso e discese che non le chiudono mai
# (il contatore si azzerava a ogni singolo delta positivo).
#
# Qui invece: quota ricampionata su griglia di distanza costante e lisciata
# su una finestra in METRI, poi algoritmo di "drawup" (minimo corrente ->
# massimo corrente) con tolleranza di discesa proporzionale al dislivello
# già accumulato, infine trim delle code piane e merge delle salite separate
# da un avvallamento breve.

def _climb_grid(df, resample_step_m=20.0, smooth_window_m=150.0):
    """Quota e tempo su griglia a passo costante di distanza orizzontale.
    Lo smoothing è espresso in metri, non in campioni: così non dipende
    dalla frequenza di registrazione dell'orologio né dalla velocità."""
    dist = np.maximum.accumulate(
        pd.to_numeric(df["distance_km"], errors="coerce").ffill().fillna(0).to_numpy() * 1000.0
    )
    ele = pd.to_numeric(df["elevation_m"], errors="coerce").ffill().bfill().fillna(0.0).to_numpy()
    tsec = pd.to_numeric(df["elapsed_sec"], errors="coerce").ffill().fillna(0).to_numpy()

    total = float(dist[-1])
    if total <= 0 or len(dist) < 3:
        return None

    n = max(3, int(total // resample_step_m))
    grid = np.linspace(0.0, total, n)
    ele_g = np.interp(grid, dist, ele)
    t_g = np.interp(grid, dist, tsec)

    w = max(3, int(round(smooth_window_m / resample_step_m)))
    if w % 2 == 0:
        w += 1
    ele_s = pd.Series(ele_g).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return grid, ele_s, t_g


def _climb_drawups(ele, min_gain_m, descent_frac, descent_max_m):
    """Candidati (i_valle, i_cima). La tolleranza di discesa cresce con il
    dislivello già fatto: 30 m di contropendenza a metà di una salita da
    1000 m non la interrompono, gli stessi 30 m su una salita da 80 m sì."""
    out = []
    i_min = i_peak = 0
    running_min = peak = float(ele[0])

    def close(a, b):
        if b > a and ele[b] - ele[a] >= min_gain_m * 0.5:
            out.append((a, b))

    for i in range(1, len(ele)):
        e = float(ele[i])
        if e > peak:
            peak, i_peak = e, i
        tol = min(max(descent_frac * (peak - running_min), 10.0), descent_max_m)
        if peak - e > tol or e < running_min:
            close(i_min, i_peak)
            running_min = peak = e
            i_min = i_peak = i

    close(i_min, i_peak)
    return out


def _climb_trim(grid, ele, a, b, min_grade_pct, win_m=100.0):
    """Toglie le code quasi piane in testa e in coda: il drawup parte dal
    minimo assoluto, che spesso cade centinaia di metri prima dell'attacco."""
    step = grid[1] - grid[0]
    w = max(1, int(round(win_m / step)))
    thr = (min_grade_pct / 100.0) * 0.5
    while b - a > 2 * w and (ele[a + w] - ele[a]) / (grid[a + w] - grid[a]) < thr:
        a += w
    while b - a > 2 * w and (ele[b] - ele[b - w]) / (grid[b] - grid[b - w]) < thr:
        b -= w
    return a, b


def _climb_merge(grid, ele, cands, max_gap_m, max_dip_m, max_dip_frac):
    """Unisce due salite separate da un breve avvallamento: un falsopiano o
    una discesa di 30 m in mezzo a un colle è parte della salita, non due
    salite distinte."""
    merged = []
    for a, b in sorted(cands):
        if merged:
            pa, pb = merged[-1]
            if a > pb:
                gap = grid[a] - grid[pb]
                dip = max(float(ele[pb] - np.min(ele[pb:a + 1])), 0.0)
                gain_tot = (ele[pb] - ele[pa]) + (ele[b] - ele[a])
                if gap <= max_gap_m and dip <= min(max_dip_m, max_dip_frac * gain_tot):
                    merged[-1] = (pa, b)
                    continue
        merged.append((a, b))
    return merged


def detect_climbs(df, min_gain_m=300.0, min_avg_grade_pct=3.0, min_length_m=800.0,
                  smooth_window_m=150.0, resample_step_m=20.0,
                  descent_frac=0.15, descent_max_m=60.0,
                  merge_gap_m=400.0, merge_dip_m=40.0, merge_dip_frac=0.15):
    """Ritorna [{'Climb Name', 'start_time', 'end_time'}, ...] in HH:MM,
    stesso contratto della versione precedente."""
    g = _climb_grid(df, resample_step_m, smooth_window_m)
    if g is None:
        return []
    grid, ele, t_g = g

    cands = _climb_drawups(ele, min_gain_m, descent_frac, descent_max_m)
    cands = [_climb_trim(grid, ele, a, b, min_avg_grade_pct) for a, b in cands]
    cands = _climb_merge(grid, ele, cands, merge_gap_m, merge_dip_m, merge_dip_frac)

    climbs = []
    for a, b in cands:
        gain = float(ele[b] - ele[a])
        length = float(grid[b] - grid[a])
        if length <= 0 or gain < min_gain_m or length < min_length_m:
            continue
        if (gain / length * 100.0) < min_avg_grade_pct:
            continue
        climbs.append({
            "Climb Name": f"Climb {len(climbs) + 1}",
            "start_time": seconds_to_hhmm(int(t_g[a])),
            "end_time": seconds_to_hhmm(int(t_g[b])),
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
            "elevation": round(float(gain), 1)
        })
    return processed

    # ---------------------------
# Climb naming & plotting helpers
# ---------------------------
# I nomi auto-generati vanno rinumerati quando l'utente cancella o riordina
# una riga: "Climb 1, Climb 3, Climb 4" è un artefatto dell'editor, non una
# scelta. I nomi scritti a mano ("Colle del Nivolet") NON si toccano.
CLIMB_AUTO_NAME_RE = re.compile(r"^\s*climb\s*\d+\s*$", re.IGNORECASE)

CLIMB_PALETTE = ["#e07b39", "#2a9d8f", "#4a7fd4", "#c0508f", "#8a63c9",
                 "#d1495b", "#3fa34d", "#c9a227", "#5c8a9e", "#a1683a"]


def renumber_climb_names(table, col="Climb Name"):
    """Riscrive in sequenza 1..N solo i nomi vuoti o del tipo 'Climb <n>'."""
    t = table.reset_index(drop=True).copy()
    if col not in t.columns:
        return t
    names = []
    for i, val in enumerate(t[col].tolist()):
        s = "" if pd.isna(val) else str(val).strip()
        names.append(f"Climb {i + 1}" if (s == "" or CLIMB_AUTO_NAME_RE.match(s)) else s)
    t[col] = names
    return t


def _hex_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_saved_climbs(df, climbs, title="Saved climbs"):
    """Profilo altimetrico con ogni salita salvata in un colore diverso.
    L'asse y parte poco sotto la quota minima invece che da zero: su una
    gara alpina il fondo a 0 m schiaccerebbe tutto il profilo in alto."""
    x = df["elapsed_sec"] / 3600.0
    ele = df["elevation_m"].rolling(window=20, min_periods=1).mean()
    lo, hi = float(np.nanmin(ele)), float(np.nanmax(ele))
    span = max(hi - lo, 1.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=ele, mode="lines", name="Profile",
        line=dict(color="rgba(140,140,140,0.85)", width=1.2),
        fill="tozeroy", fillcolor="rgba(140,140,140,0.10)",
        hovertemplate="%{x:.2f} h — %{y:.0f} m<extra></extra>",
    ))

    for i, c in enumerate(climbs):
        s, e = c.get("start_idx"), c.get("end_idx")
        if s is None or e is None:
            s_sec, e_sec = hhmm_to_seconds(c.get("start_time")), hhmm_to_seconds(c.get("end_time"))
            if s_sec is None or e_sec is None:
                continue
            s, e = nearest_idx(df, s_sec), nearest_idx(df, e_sec)
        seg_x, seg_y = x.iloc[s:e + 1], ele.iloc[s:e + 1]
        if seg_x.empty:
            continue

        col = CLIMB_PALETTE[i % len(CLIMB_PALETTE)]
        name = c.get("name", f"Climb {i + 1}")
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", name=name,
            line=dict(color=col, width=2.4),
            fill="tozeroy", fillcolor=_hex_rgba(col, 0.30),
            hovertemplate=(f"<b>{name}</b><br>%{{x:.2f}} h — %{{y:.0f}} m"
                           "<extra></extra>"),
        ))
        # etichetta sulla cima: la legenda da sola costringe a fare
        # avanti-indietro tra colore e nome
        i_top = int(seg_y.idxmax())
        fig.add_annotation(
            x=float(x.loc[i_top]), y=float(ele.loc[i_top]),
            text=f"<b>{name}</b>", showarrow=False, yshift=12,
            font=dict(size=11, color=col),
            bgcolor="rgba(255,255,255,0.70)", borderpad=2,
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Elapsed Time (hours)",
                   showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                   showline=True, linecolor="rgba(128,128,128,0.65)",
                   mirror=True, ticks="outside", ticklen=4),
        yaxis=dict(title="Elevation (m)", tickformat="d",
                   range=[lo - 0.06 * span, hi + 0.18 * span],
                   showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                   showline=True, linecolor="rgba(128,128,128,0.65)",
                   mirror=True, ticks="outside", ticklen=4),
        hovermode="x unified", height=440, margin=dict(t=70),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

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
        index=1,
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

            cA, cB, cC = st.columns(3)
            min_gain_m = cA.number_input(
                "Min elevation gain (m)", value=300, min_value=20, step=25,
                key="climb_min_gain")
            min_grade = cB.number_input(
                "Min average gradient (%)", value=3.0, min_value=0.0, max_value=30.0,
                step=0.5, key="climb_min_grade")
            smooth_m = cC.number_input(
                "Profile smoothing (m)", value=150, min_value=20, max_value=1000,
                step=25, key="climb_smooth",
                help="Più alto = ignora le micro-ondulazioni. Alza se ottieni "
                     "troppe salite spezzettate.")

            with st.expander("⚙️ Advanced detection settings"):
                a1, a2, a3 = st.columns(3)
                min_len_m = a1.number_input(
                    "Min climb length (m)", value=800, min_value=100, step=100,
                    key="climb_min_len")
                merge_gap = a2.number_input(
                    "Merge gap (m)", value=400, min_value=0, step=100,
                    key="climb_merge_gap",
                    help="Due salite più vicine di così vengono unite.")
                merge_dip = a3.number_input(
                    "Max dip when merging (m)", value=40, min_value=0, step=10,
                    key="climb_merge_dip")

            @st.cache_data(show_spinner="Detecting climbs...")
            def cached_detect_climbs(_df, min_gain_m, min_grade, smooth_m,
                                     min_len_m, merge_gap, merge_dip):
                return detect_climbs(
                    _df, min_gain_m=min_gain_m, min_avg_grade_pct=min_grade,
                    min_length_m=min_len_m, smooth_window_m=smooth_m,
                    merge_gap_m=merge_gap, merge_dip_m=merge_dip,
                )

            raw_climbs = cached_detect_climbs(
                df, min_gain_m, min_grade, smooth_m, min_len_m, merge_gap, merge_dip)
            
            if not raw_climbs:
                st.warning("No climbs detected with current parameters.")
            else:
                temp_climbs = add_temporary_metrics(df, [c.copy() for c in raw_climbs])

                # La tabella si ricostruisce quando cambiano i parametri di
                # detection, altrimenti le modifiche manuali dell'utente
                # andrebbero perse a ogni rerun.
                _sig = (min_gain_m, min_grade, smooth_m, min_len_m,
                        merge_gap, merge_dip, len(raw_climbs))
                if (st.session_state.get("_climb_detect_sig") != _sig
                        or "editable_climb_table" not in st.session_state):
                    editable_df = pd.DataFrame(temp_climbs)
                    cols_desired = ["Climb Name", "start_time", "end_time",
                                    "distance_km", "elev_gain_m"]
                    for col in cols_desired:
                        if col not in editable_df.columns:
                            editable_df[col] = ""
                    st.session_state["editable_climb_table"] = editable_df[cols_desired]
                    st.session_state["_climb_detect_sig"] = _sig

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
                    st.info("You can edit climb names, start times, and end times. "
                            "Auto-generated names are renumbered when you add or "
                            "delete a row; custom names are left alone.")

                    # La key dell'editor porta un contatore: quando i nomi
                    # vengono rinumerati bisogna forzare un remount, altrimenti
                    # st.data_editor riapplica il suo diff interno sul vecchio
                    # dataframe e i nomi tornano come prima.
                    ver = st.session_state.get("_climb_editor_ver", 0)
                    edited = st.data_editor(
                        st.session_state["editable_climb_table"],
                        num_rows="dynamic",
                        key=f"climb_editor_{ver}",
                        disabled=["distance_km", "elev_gain_m"]
                    )

                    renamed = renumber_climb_names(edited)
                    before = ["" if pd.isna(v) else str(v).strip()
                              for v in edited.get("Climb Name", [])]
                    after = list(renamed["Climb Name"])
                    st.session_state["editable_climb_table"] = renamed

                    if before != after:
                        st.session_state["_climb_editor_ver"] = ver + 1
                        st.rerun(scope="fragment")

                climb_table_fragment()

                if st.button("Save climbs and compute metrics", key="save_climbs_auto"):
                    try:
                        _tbl = renumber_climb_names(st.session_state["editable_climb_table"])
                        st.session_state["editable_climb_table"] = _tbl
                        processed_climbs = compute_processed_climbs(df, _tbl)
                        st.session_state["climb_data"] = processed_climbs
                        st.success("✅ Climbs saved and metrics computed!")
                    except ValueError as e:
                        st.error(str(e))

                if st.session_state.get("climb_data"):
                    st.markdown("#### ✅ Saved climbs")
                    st.plotly_chart(
                        plot_saved_climbs(df, st.session_state["climb_data"],
                                          title="Saved climbs — elevation profile"),
                        use_container_width=True,
                    )

                    final_df = pd.DataFrame(st.session_state["climb_data"])
                    display_cols = ["name", "start_time", "end_time",
                                    "duration", "distance", "elevation"]
                    st.dataframe(
                        final_df[[c for c in display_cols if c in final_df.columns]]
                        .rename(columns={
                            "name": "Climb", "start_time": "Start",
                            "end_time": "End", "duration": "Duration",
                            "distance": "Distance (km)", "elevation": "D+ (m)",
                        }),
                        use_container_width=True, hide_index=True,
                    )
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
                st.session_state[table_key] = pd.DataFrame(columns=["name", "start_time", "end_time"])

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

                # --- CSV import ---
                st.markdown("#### 📥 Import Laps from CSV")
                imported_lap_csv = st.file_uploader(
                    "Import previously exported lap CSV:",
                    type=["csv"],
                    key="import_lap_csv"
                )
                if imported_lap_csv is not None:
                    try:
                        imported_df = pd.read_csv(imported_lap_csv)
                        required_import_cols = {"name", "start_km", "end_km"}
                        if not required_import_cols.issubset(set(imported_df.columns)):
                            st.error(f"⚠️ CSV must contain columns: name, start_km, end_km")
                        else:
                            imported_df = imported_df[["name", "start_km", "end_km"]]
                            imported_df["ngp"] = ""
                            imported_df["name"] = imported_df["name"].fillna("").astype(str)
                            st.session_state["manual_lap_table"] = imported_df.reset_index(drop=True)
                            st.success(f"✅ Imported {len(imported_df)} lap(s) from CSV!")
                    except Exception as e:
                        st.error(f"⚠️ Error reading CSV: {e}")

                # --- Editable table ---
                table_key = "manual_lap_table"
                if table_key not in st.session_state:
                    st.session_state[table_key] = pd.DataFrame({
                        "name":     pd.Series([], dtype="str"),
                        "start_km": pd.Series([], dtype="float"),
                        "end_km":   pd.Series([], dtype="float"),
                        "ngp":      pd.Series([], dtype="str"),
                    })

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

                    # --- Auto-name any unnamed rows first ---
                    edited["name"] = [
                        row["name"] if pd.notna(row.get("name")) and str(row.get("name", "")).strip() != ""
                        else f"Lap {i+1}"
                        for i, row in edited.iterrows()
                    ]

                    # --- Check race name is available for export ---
                    race_name_ok = bool(st.session_state.get("race_name", "").strip())
                    if not race_name_ok:
                        st.warning("⚠️ Race name is not set — please submit the Race Info form above to enable CSV export.")

                    # --- Validation: start_km and end_km required ---
                    incomplete_rows = []
                    for i, row in edited.iterrows():
                        if pd.isna(row.get("start_km")):
                            incomplete_rows.append(f"Row {i+1} ({row['name']}): missing Start (km)")
                        if pd.isna(row.get("end_km")):
                            incomplete_rows.append(f"Row {i+1} ({row['name']}): missing End (km)")
                        elif not pd.isna(row.get("start_km")) and float(row["end_km"]) <= float(row["start_km"]):
                            incomplete_rows.append(f"Row {i+1} ({row['name']}): End km must be greater than Start km")

                    if incomplete_rows:
                        for msg in incomplete_rows:
                            st.error(f"⚠️ {msg}")
                        st.stop()

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
                        st.session_state["lap_export_ready"] = True
                        st.session_state["lap_export_df"] = edited[["name", "start_km", "end_km"]].copy()
                        st.success(f"✅ {len(lap_data)} lap(s) saved!")
                    else:
                        st.session_state["lap_export_ready"] = False
                        st.error("⚠️ No valid laps to save. Please check your distance values.")

                # --- CSV export (outside form) ---
                if st.session_state.get("lap_export_ready"):
                    race_name_ok = bool(st.session_state.get("race_name", "").strip())
                    if race_name_ok:
                        csv_bytes = st.session_state["lap_export_df"].to_csv(index=False).encode("utf-8")
                        safe_race_name = st.session_state["race_name"].strip().replace(" ", "_")
                        st.download_button(
                            label="📥 Download Laps as CSV",
                            data=csv_bytes,
                            file_name=f"{safe_race_name}_lap_analyzer.csv",
                            mime="text/csv",
                            key="download_lap_csv"
                        )
                    else:
                        st.info("ℹ️ Submit the Race Info form above to enable CSV export.")
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
# --- Archiviazione file grezzi via email ---------------------------------
# Nessun invio automatico: parte solo al click. Il bottone resta disabilitato
# finché non ci sono sia il file sia i dati di gara: un FIT che arriva senza
# nome atleta è inutilizzabile in archivio.
if uploaded_file is not None:
    _missing = missing_archive_requirements()

    st.markdown(
        """
        <div style="background-color:#fff3cd; padding:10px; border-left:5px solid #e0a800; border-radius:5px; color:#7a5c00; margin-bottom:10px;">
        ⚠️ If you wanna help the development of this app, please consider sending the developer the race and cardiac data you're using on this analysis clicking the button below ⚠️
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "You can choose between sending the full data or an anonymized version. The full data will "
        "be stored in the developer's archive for future analysis and development, while the "
        "'anonymized' version will be used for testing and improving the app without any personal information."
    )

    _send_col, _spacer = st.columns([2, 4])
    _subject = (f"[RACE ANALYZER] App data for {st.session_state.get('athlete_name', '?')} - "
                f"{st.session_state.get('race_name', '?')}")
    _body = (f"Athlete: {st.session_state.get('athlete_name', '')}\n"
             f"Race: {st.session_state.get('race_name', '')}\n"
             f"Distance: {st.session_state.get('kilometers', '?')} km\n"
             f"D+: {st.session_state.get('total_elevation_gain', '?')} m\n")

    _clicked = _send_col.button(
        "📧 Send raw files by email",
        key="send_files_email",
        disabled=bool(_missing),
        help=("Missing: " + ", ".join(_missing)) if _missing else
             "Sends the raw .fit file and the HR zones CSV",
    )

    if _missing:
        st.caption("⚠️ Before sending, please complete: " + ", ".join(_missing))

    if _clicked:
        # Ricontrollo al click: fra il rendering del bottone e la pressione
        # può esserci stato un rerun che ha svuotato qualcosa.
        _missing = missing_archive_requirements()
        if _missing:
            st.error("❌ Missing data: " + ", ".join(_missing))
        else:
            try:
                _payload = build_archive_payload(uploaded_file.getvalue(), uploaded_file.name)
                _n = send_payload_email(_payload, _subject, _body)
                st.success(f"✅ Email sent with {_n} attachment(s).")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

        
    # --- Invio anonimo ---------------------------------------------------
    # Stessi allegati, ma senza alcun riferimento all'atleta: filename
    # generici e corpo del messaggio vuoto. Il bottone è separato invece
    # di essere una checkbox perché la scelta va fatta consapevolmente.
    _anon_missing = missing_anonymous_requirements()
    _anon_col, _anon_spacer = st.columns([2, 4])

    _anon_clicked = _anon_col.button(
        "🕶️ Send anonymized data",
        key="send_files_email_anon",
        disabled=bool(_anon_missing),
        help=("Missing: " + ", ".join(_anon_missing)) if _anon_missing else
             "Sends the same files with no name, race or date attached",
    )

    if _anon_clicked:
        _anon_missing = missing_anonymous_requirements()
        if _anon_missing:
            st.error("❌ Missing data: " + ", ".join(_anon_missing))
        else:
            try:
                _payload = build_archive_payload(
                    uploaded_file.getvalue(), uploaded_file.name, anonymous=True)
                _n = send_payload_email(_payload, "[RACE ANALYZER] Anonymous data", "")
                st.success(f"✅ Anonymous data sent ({_n} attachment(s)).")
            except Exception as e:
                st.error(f"❌ Email failed: {e}")

if not analysis_ready:
    if uploaded_file is None:
        st.info("👆 Upload a FIT file to run the analysis")
    else:
        st.warning(
            "🔒 **Analysis locked** — press **Submit HR Zones** above to confirm "
            "the athlete's heart rate zones."
        )
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

if analysis_ready:

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
# -------------------------#
# ---- LIVE CHARTS ------------- #

if analysis_ready:

    # --- Prepare hover info ---
    df_clean["Race Time [h:mm]"] = df_clean.apply(
        lambda row: f"{int(row['elapsed_sec']//3600)}:{int((row['elapsed_sec']%3600)//60):02d} | HR: {int(row['hr_smooth'])} bpm",
        axis=1
    )

    efs_df, efs_totals = compute_efs_series(df)



    st.markdown(
        "<p style='text-align:center; margin-bottom:0.2em;'>"
        "What would you like to display on the trend graph?</p>",
        unsafe_allow_html=True
    )
    # Le colonne esterne fanno da margine: le due checkbox restano vicine
    # al centro invece di finire ai bordi opposti della pagina.
    _, c_hr, c_efs, c_ele, _ = st.columns([1, 2, 2.4, 2, 1])
    show_hr = c_hr.checkbox("❤️ Heart Rate", value=True, key="live_show_hr")
    show_efs = c_efs.checkbox("🏃 Equivalent Flat Speed", value=True, key="live_show_efs")
    show_ele = c_ele.checkbox("⛰️ Elevation", value=True, key="live_show_ele")
    st.write("")

    # Rolling median sui segmenti: l'EFS grezza a 20 m è troppo rumorosa per
    # essere letta a occhio. Si interpola prima sui NaN (soste) così la
    # finestra non si svuota e la linea resta continua.
    if efs_df is not None:
        efs_df["efs_smooth"] = (
            efs_df["efs_kmh"]
            .interpolate(limit_direction="both")
            .rolling(EFS_PLOT_SMOOTH_PTS, center=True,
                     min_periods=max(3, EFS_PLOT_SMOOTH_PTS // 4))
            .median()
        )

    fig = go.Figure()

    if show_hr:
        fig.add_trace(go.Scatter(
            x=df_clean["elapsed_hours"], y=df_clean["hr_smooth"],
            mode="lines", name="Heart Rate",
            line=dict(color="rgba(70,140,220,1)", width=1.6),
            customdata=df_clean["Race Time [h:mm]"],
            hovertemplate="%{customdata}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=df_clean["elapsed_hours"], y=df_clean["trend_line"],
            mode="lines", name="HR Trend",
            line=dict(color="red", dash="dash", width=2),
            hoverinfo="skip",
        ))

    efs_decay_pct_h = None
    spd_index = spd_index_str = spd_kmh_per_h = None
    if efs_df is not None:
        efs_valid = efs_df.dropna(subset=["efs_kmh"])

        if show_efs:
            fig.add_trace(go.Scatter(
                x=efs_df["elapsed_hours"], y=efs_df["efs_smooth"],
                mode="lines", name="EFS", yaxis="y2",
                line=dict(color="rgba(42,157,143,1)", width=1.8),
                customdata=efs_df["km"],
                hovertemplate="EFS: %{y:.2f} km/h<br>Km %{customdata:.2f}<extra></extra>",
            ))

        if len(efs_valid) > 2:
            Xe = efs_valid["elapsed_hours"].values.reshape(-1, 1)
            ye = efs_valid["efs_kmh"].values
            reg_efs = LinearRegression().fit(Xe, ye)

            if show_efs:
                fig.add_trace(go.Scatter(
                    x=efs_valid["elapsed_hours"], y=reg_efs.predict(Xe),
                    mode="lines", name="EFS Trend", yaxis="y2",
                    line=dict(color="mediumspringgreen", dash="dash", width=2),
                    hoverinfo="skip",
                ))

            slope_efs = float(reg_efs.coef_[0])
            efs_at_start = float(reg_efs.intercept_)
            if efs_at_start > 0:
                efs_decay_pct_h = -slope_efs / efs_at_start * 100

            slope_efs = float(reg_efs.coef_[0])
            efs_at_start = float(reg_efs.intercept_)
            if efs_at_start > 0:
                efs_decay_pct_h = -slope_efs / efs_at_start * 100

            # slope_efs è già in km/h per ORA (x = elapsed_hours).
            # SPD usa la pendenza al secondo × 5e4: su 58 gare questa scala
            # dà a SPD la stessa distribuzione del DET (mediana ~4.5,
            # p90 ~12), quindi le soglie 4/7/10 valgono per entrambi.
            # Stessa costante cosmetica del DET (1e4) sulla pendenza al
            # secondo. Le soglie SPD sono quelle del DET divise per 5,
            # perché su 58 gare la pendenza della velocità è circa un
            # quinto di quella della FC in valore assoluto.
            spd_kmh_per_h = -slope_efs              # positivo = rallenta
            spd_index = abs(slope_efs / 3600.0) * SPD_INDEX_SCALE
            spd_index_str = f"{spd_index:.2f}"

    # Profilo altimetrico su terzo asse, disegnato per primo nello z-order
    # naturale di Plotly? No: le tracce aggiunte dopo stanno sopra. Qui va
    # bene, è un riempimento chiaro e le curve restano leggibili sopra.
    if show_ele:
        _ele_x = df["elapsed_sec"] / 3600.0
        _ele_y = df["elevation_m"].rolling(window=20, min_periods=1).mean()
        fig.add_trace(go.Scatter(
            x=_ele_x, y=_ele_y,
            mode="lines", name="Elevation", yaxis="y3",
            line=dict(color="rgba(150,150,150,0.9)", width=1),
            fill="tozeroy", fillcolor="rgba(150,150,150,0.18)",
            hovertemplate="Elevation: %{y:.0f} m<extra></extra>",
        ))

    # Con entrambe le curve accese si allarga il range dell'asse EFS: la
    # curva viene schiacciata nella metà bassa e non si incrocia con la FC.
    # Con una sola curva il range è naturale, altrimenti si leggerebbe
    # schiacciata senza motivo.
    y2_range = None
    if show_efs and efs_df is not None and efs_df["efs_smooth"].notna().any():
        efs_top = float(efs_df["efs_smooth"].max())
        y2_range = [0, efs_top * (EFS_AXIS_HEADROOM if show_hr else 1.15)]

    y1_range = None
    if show_hr and show_efs and df_clean["hr_smooth"].notna().any():
        hr_lo = float(df_clean["hr_smooth"].min())
        hr_hi = float(df_clean["hr_smooth"].max())
        span = max(hr_hi - hr_lo, 1.0)
        y1_range = [hr_lo - 0.45 * span, hr_hi + 0.05 * span]

    # Terzo asse: nascosto (l'utente legge le quote dall'hover) e schiacciato
    # in basso, altrimenti il profilo coprirebbe le due curve. Mostrarlo
    # richiederebbe un domain ristretto sull'asse x e ruberebbe larghezza.
    y3_range = None
    if show_ele:
        _e_hi = float(df["elevation_m"].max())
        _e_lo = float(df["elevation_m"].min())
        _e_span = max(_e_hi - _e_lo, 1.0)
        y3_range = [_e_lo - 0.35 * _e_span, _e_hi + (ELE_AXIS_HEADROOM - 1) * _e_span]

    _title = "Heart Rate & Equivalent Flat Speed Over Time"
    if show_ele:
        _title += " (with elevation)"

    # Ripartizione verticale: HR 50%, EFS 30%, elevazione 20% quando tutte
    # e tre sono accese. Le fasce vengono ricalcolate in base a cosa è
    # effettivamente visibile, così una sola serie usa tutta l'altezza.
    _wanted = [(show_hr, 0.42), (show_efs, 0.28), (show_ele, 0.30)]
    _active = [w for on, w in _wanted if on]
    _tot = sum(_active) or 1.0
    _gaps = [0.012, 0.05]   # [HR-EFS, EFS-elevazione]

    # Gap variabile per coppia: FC ed EFS quasi attaccate (si leggono
    # insieme), profilo staccato perché è contesto, non una curva da
    # confrontare punto per punto. Si contano solo i gap tra bande
    # ENTRAMBE visibili, altrimenti spegnendo una serie resta un buco.
    _used_gaps = []
    _prev_on = None
    for i, (on, _w) in enumerate(_wanted):
        if on and _prev_on is not None:
            _used_gaps.append(_gaps[min(_prev_on, len(_gaps) - 1)])
        if on:
            _prev_on = i
    _gap_total = sum(_used_gaps)

    _doms, _top, _gi = [], 1.0, 0
    for on, w in _wanted:
        if not on:
            _doms.append([0.0, 1.0])
            continue
        h = (w / _tot) * (1.0 - _gap_total)
        _doms.append([max(_top - h, 0.0), _top])
        _top -= h
        if _gi < len(_used_gaps):
            _top -= _used_gaps[_gi]
            _gi += 1
    _dom_hr, _dom_efs, _dom_ele = _doms

        # asse y della banda più bassa attiva: è lì che va appoggiata la x
    _x_anchor = "y"
    if show_ele:
        _x_anchor = "y3"
    elif show_efs:
        _x_anchor = "y2"

    fig.update_layout(
        title=_title,
        # Bande orizzontali separate invece di tre assi sovrapposti: con
        # `domain` ogni serie ha la sua fascia e non può accavallarsi.
        # Le bande si allargano a riempire lo spazio delle serie spente.
        # L'asse x si ancora alla banda PIÙ BASSA visibile, altrimenti
        # Plotly lo disegna sotto la fascia della FC, cioè a metà figura.
        # Ogni banda incorniciata come un subplot matplotlib: linee di
        # contorno su tutti e quattro i lati e assi y tutti a sinistra.
        # È la cornice, più della griglia, a rendere leggibili tre serie
        # impilate — l'occhio ha un contenitore per ciascuna.
        xaxis=dict(title="Elapsed Time (hours)", anchor=_x_anchor,
                   showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                   showline=True, linecolor="rgba(128,128,128,0.65)",
                   mirror=True, ticks="outside", ticklen=4),
        yaxis=dict(title="Heart Rate (bpm)", tickformat="d",
                   domain=_dom_hr,
                   showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                   showline=True, linecolor="rgba(128,128,128,0.65)",
                   mirror=True, ticks="outside", ticklen=4),
        yaxis2=dict(title="EFS (km/h)", domain=_dom_efs, anchor="x",
                    showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                    showline=True, linecolor="rgba(128,128,128,0.65)",
                    mirror=True, ticks="outside", ticklen=4),
        yaxis3=dict(title="Elev (m)", domain=_dom_ele, anchor="x",
                    showgrid=True, gridcolor="rgba(128,128,128,0.20)",
                    showline=True, linecolor="rgba(128,128,128,0.65)",
                    mirror=True, ticks="outside", ticklen=4),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=90),
        height=600,
    )

    if not show_hr and not show_efs and not show_ele:
        st.info("Select at least one curve to display.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    # --- Indici di decadimento: FC sopra, velocità sotto ---
    # Ogni riga ha lo stesso schema: badge dell'indice a sinistra, tasso
    # grezzo a destra nello stesso stile. Segno incluso: negativo = perde,
    # positivo = guadagna.
    def _rate_box(value, unit, box_color):
        return (f"<span style='font-size:16px; background-color:{box_color}; color:black; "
                f"padding:5px; border-radius:5px; display:inline-block;'>"
                f"<b>{value:+.3f}</b> {unit}</span>")

    st.markdown("**Il DET index indica il decadimento della FC nel corso del tempo**")
    det_tooltip = ("DI < 4 - SCARSO DECADIMENTO\n"
                   "DI = 7 - DECADIMENTO MEDIO\n"
                   "DI > 10 - ALTO DECADIMENTO")

    _hr_bpm_per_h = float(reg.coef_[0]) * 3600.0
    st.markdown(
        "<div style='display:flex; gap:12px; align-items:center;'>"
        f"<span title='{det_tooltip}' style='font-size:16px; background-color:{color}; "
        f"color:black; padding:5px; border-radius:5px;'>"
        f"📈 DET INDEX: <b>{det_index_str}</b> ({comment})</span>"
        + _rate_box(_hr_bpm_per_h, "bpm per ora", color) +
        "</div>",
        unsafe_allow_html=True)

    st.write(" ")
    st.markdown("**L'EFS è la velocità normalizzata per la pendenza:**")

    if spd_index_str is not None:
        # Soglie DET / 5: la pendenza della velocità è circa un quinto
        # di quella della FC in valore assoluto (58 gare).
        if spd_index < 0.8:
            spd_comment, spd_color = "Scarso decadimento", "green"
        elif spd_index <= 2.0:
            spd_comment, spd_color = "Decadimento medio", "cyan"
        else:
            spd_comment, spd_color = "Alto decadimento", "lightcoral"
        spd_tooltip = ("SPD < 0.8 - SCARSO DECADIMENTO\n"
                       "SPD = 1.4 - DECADIMENTO MEDIO\n"
                       "SPD > 2.0 - ALTO DECADIMENTO")

        st.markdown(
            "<div style='display:flex; gap:12px; align-items:center;'>"
            f"<span title='{spd_tooltip}' style='font-size:16px; background-color:{spd_color}; "
            f"color:black; padding:5px; border-radius:5px;'>"
            f"📉 SPD INDEX: <b>{spd_index_str}</b> ({spd_comment})</span>"
            + _rate_box(-spd_kmh_per_h, "km/h per ora", spd_color) +
            "</div>",
            unsafe_allow_html=True)

    # =====================================================================
    # EFFICIENCY FACTOR — velocità per battito
    # =====================================================================
    # EF = EFS / FC, in metri all'ora per battito. Valore assoluto, non
    # normalizzato: è un livello leggibile di suo, e il suo calo dice
    # quanto costa in battiti la stessa velocità equivalente.
    # Il ×1000 è solo cosmetico: in km/h per bpm verrebbero numeri tipo
    # 0.045, in m/h per bpm la stessa grandezza sta sui 40-60.
    EF_WIN_MIN = 60      # finestra mobile (minuti)
    EF_STEP_MIN = 15      # passo tra finestre

    if efs_df is not None and efs_df["efs_kmh"].notna().any():
        st.divider()
        st.markdown("### Efficiency Factor — velocità per sforzo relativo")
        
        _hr_h = df_clean["elapsed_hours"].to_numpy()
        _hr_v = df_clean["hr_smooth"].to_numpy()
        _ef_x, _ef_hr, _ef_efs = [], [], []

        _w = EF_WIN_MIN / 60.0
        _t = max(_hr_h.min(), efs_df["elapsed_hours"].min()) + _w / 2
        _t_end = min(_hr_h.max(), efs_df["elapsed_hours"].max()) - _w / 2
        while _t <= _t_end:
            _mh = (_hr_h >= _t - _w / 2) & (_hr_h <= _t + _w / 2)
            _me = ((efs_df["elapsed_hours"] >= _t - _w / 2)
                   & (efs_df["elapsed_hours"] <= _t + _w / 2))
            if _mh.sum() >= 10 and _me.sum() >= 5:
                _hrm = float(np.nanmean(_hr_v[_mh]))
                _efm = float(efs_df.loc[_me, "efs_kmh"].mean())
                if _hrm > 0 and np.isfinite(_efm):
                    _ef_x.append(_t)
                    _ef_hr.append(_hrm)
                    _ef_efs.append(_efm)
            _t += EF_STEP_MIN / 60.0

        if len(_ef_x) < 4:
            st.info("Traccia troppo corta per l'analisi a finestra mobile.")
        else:
            _ef_x = np.array(_ef_x)
            _ef_hr = np.array(_ef_hr)
            _ef_efs = np.array(_ef_efs)
            # FC in FRAZIONE della soglia, non in bpm: la FC di soglia è
            # in larga parte genetica e non dice nulla su velocità o
            # fitness. Dividendo per z4 due atleti alla stessa velocità
            # relativa alla propria soglia ottengono lo stesso EF, e il
            # numero diventa confrontabile tra atleti.
            # EF = km/h per unità di sforzo relativo (1.0 = a soglia).
            _hr_thr = float(st.session_state.get("z4", 0) or 0)
            _ef_rel = _ef_hr / _hr_thr if _hr_thr > 0 else None
            _ef = _ef_efs / _ef_rel if _ef_rel is not None else None

            def _pad(arr, frac=0.10):
                """Range con margine: con tre assi sovrapposti Plotly
                autoscala ciascuno per conto suo e taglia i picchi."""
                lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                span = max(hi - lo, 1e-9)
                return [lo - frac * span, hi + frac * span]

            if _ef is None:
                st.warning(
                    "⚠️ Imposta la FC di soglia (Zona 4) nella sezione "
                    "**Athlete Heart Rate Zones** per calcolare l'Efficiency Factor."
                )
                st.stop()

            fig_ef = go.Figure()

            # Profilo altimetrico di sfondo su un quarto asse invisibile,
            # schiacciato in basso: serve solo a dare contesto ai cali di
            # EF, non è una curva da leggere. Aggiunto per PRIMO perché
            # Plotly disegna nell'ordine di inserimento.
            _bg_x = df["elapsed_sec"] / 3600.0
            _bg_y = df["elevation_m"].rolling(window=20, min_periods=1).mean()
            fig_ef.add_trace(go.Scatter(
                x=_bg_x, y=_bg_y, mode="lines", name="Elevation",
                yaxis="y4", showlegend=False,
                line=dict(color="rgba(150,150,150,0.45)", width=0.8),
                fill="tozeroy", fillcolor="rgba(150,150,150,0.10)",
                hoverinfo="skip",
            ))
            fig_ef.add_trace(go.Scatter(
                x=_ef_x, y=_ef, mode="lines", name="EF (km/h a soglia)",
                line=dict(color="#e07b39", width=2.6),
                hovertemplate="EF: %{y:.2f} km/h per unità di sforzo<extra></extra>",
            ))
            fig_ef.add_trace(go.Scatter(
                x=_ef_x, y=_ef_rel * 100.0, mode="lines",
                name="FC (% soglia)", yaxis="y2",
                line=dict(color="rgba(70,140,220,0.6)", width=1.4),
                hovertemplate="FC: %{y:.0f}% della soglia<extra></extra>",
            ))
            fig_ef.add_trace(go.Scatter(
                x=_ef_x, y=_ef_efs, mode="lines", name="EFS (km/h)", yaxis="y3",
                line=dict(color="rgba(42,157,143,0.6)", width=1.4),
                hovertemplate="EFS: %{y:.2f} km/h<extra></extra>",
            ))

            fig_ef.update_layout(
                title=f"Efficiency Factor (finestra {EF_WIN_MIN} min) — FC relativa alla soglia",
                xaxis=dict(title="Elapsed Time (hours)", domain=[0.0, 0.88],
                           hoverformat=".2f"),
                yaxis=dict(title="EF (km/h per sforzo relativo)", range=_pad(_ef)),
                yaxis2=dict(title="FC (% soglia)", overlaying="y", side="right",
                            showgrid=False, range=_pad(_ef_rel * 100.0)),
                yaxis3=dict(title="EFS (km/h)", overlaying="y", side="right",
                            position=0.97, showgrid=False, range=_pad(_ef_efs)),
                yaxis4=dict(overlaying="y", side="right", visible=False,
                            range=[float(_bg_y.min()), float(_bg_y.max()) * 2.6]),
                height=460, hovermode="x unified",
                margin=dict(t=80),
                legend=dict(orientation="h", yanchor="bottom", y=1.04,
                            xanchor="left", x=0),
            )
            st.plotly_chart(fig_ef, use_container_width=True)

            _half = len(_ef) // 2
            _ef_1, _ef_2 = _ef[:_half].mean(), _ef[_half:].mean()
            _ef_delta = _ef_2 - _ef_1
            _ef_cmt, _ef_col = ef_delta_verdict(_ef_delta)

            # Il delta ha lo stesso peso visivo degli altri tre valori, non
            # più il numerino sotto la metrica: è la variabile che conta.
            st.markdown(
                "<div style='display:flex; gap:12px; align-items:stretch; "
                "flex-wrap:wrap; margin-bottom:6px;'>"
                + ef_metric_box("EF 1ª metà", f"{_ef_1:.2f}")
                + ef_metric_box("EF 2ª metà", f"{_ef_2:.2f}")
                + ef_metric_box("EF medio gara", f"{_ef.mean():.2f}")
                + ef_metric_box(f"Δ EF — {_ef_cmt}", f"{_ef_delta:+.2f}",
                                _ef_col, "black")
                + "</div>",
                unsafe_allow_html=True)

            st.caption(
                f"EF = EFS ÷ (FC / soglia). Con soglia a {_hr_thr:.0f} bpm, "
            )

# ------------------------------------------
# SEGMENT ANALYSIS #
# ------------------------------------------

    st.markdown("## SEGMENT ANALYSIS")

    # Time in zone analysis for segments

    # Check conditions

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
if analysis_ready and 'df' in locals() and not df.empty:

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
if analysis_ready and 'HR Zone' in df.columns:
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

if analysis_ready and 'HR Zone' in df.columns and all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):

    st.markdown("### 📊 Heart Rate Density Distribution by Zone")

    # --- Zone boundaries ---
    _zone_bands = [
        {"name": "Z1", "x0": 0,   "x1": z1, "color": "rgba(100, 200, 255, 0.15)"},
        {"name": "Z2", "x0": z1, "x1": z2, "color": "rgba(100, 220, 100, 0.15)"},
        {"name": "Z3", "x0": z2, "x1": z3, "color": "rgba(255, 230, 50,  0.15)"},
        {"name": "Z4", "x0": z3, "x1": z4, "color": "rgba(255, 150, 50,  0.15)"},
        {"name": "Z5", "x0": z4, "x1": z5, "color": "rgba(255, 80,  80,  0.15)"},
    ]

    DENSITY_CHART_HEIGHT = 160   # abbassa per stringere ancora
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

        # Titolo dentro l'area del grafico e niente titolo asse x: sono i due
        # elementi che costano più spazio verticale, e con i grafici impilati
        # quello spazio è tutto scroll in più. L'annotazione va in alto a
        # sinistra, dove la densità è quasi sempre bassa.
        fig.add_annotation(
            x=0.01, y=0.97, xref="paper", yref="paper",
            text=f"<b>{title}</b>", showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="rgba(255,255,255,0.75)",
            xanchor="left", yanchor="top",
        )

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Density %",
            showlegend=False,
            plot_bgcolor="white",
            height=DENSITY_CHART_HEIGHT,
            margin=dict(t=14, b=24, l=46, r=12),
            xaxis=dict(range=[x_min, x_max]),
            font=dict(size=11),
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

    # Impilati, non affiancati: le distribuzioni si confrontano guardando
    # dove cade il picco sulla stessa scala x, e con due grafici a metà
    # larghezza separati da uno stacco l'occhio non ci riesce.
    for _start_sec, _end_sec, _seg_name in _segment_inputs:
        _df_seg = df[(df["elapsed_sec"] >= _start_sec) & (df["elapsed_sec"] <= _end_sec)]
        _hr_seg = _df_seg["heart_rate"].dropna()
        if len(_hr_seg) > 1:
            st.plotly_chart(
                build_density_chart(
                    _hr_seg,
                    f"HR Density - {_seg_name}",
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

# `bar_df` esiste solo se le zone FC sono state inserite e il blocco del
# grafico a barre è stato eseguito: va verificata l'ESISTENZA del nome, non
# il suo valore, altrimenti NameError invece del warning in fondo.
if (uploaded_file is not None
        and 'bar_df' in locals()
        and bar_df is not None
        and not bar_df.empty):

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
                efs_climb = compute_segment_efs(df_seg)
                efs_str   = f"{efs_climb:.2f}" if pd.notna(efs_climb) else "-"
                rows.append([
                    entry.get("name", "Climb"),
                    duration_hms, distance, elevation,
                    avg_fc, avg_grade, vam, efs_str
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
            extra_cols = ["Avg FC", "Avg Grade (%)", "VAM (m/h)", "Avg EFS (km/h)"]
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

                # --- TOM LOW SECRET DATA ---
                if "show_tom_low" not in st.session_state:
                    st.session_state["show_tom_low"] = False

                if st.button("TOM LOW SECRET DATA 🔒", key="tom_low_btn"):
                    st.session_state["show_tom_low"] = not st.session_state["show_tom_low"]

                if st.session_state.get("show_tom_low"):
                    tom_rows = []
                    for _, row in climb_df.iterrows():
                        name = row.get("Climb Name", "")
                        try:
                            avg_hr = float(str(row.get("Avg FC", 0)).replace(",", ".") or 0)
                            grade  = float(str(row.get("Avg Grade (%)", 0)).replace(",", ".") or 0)
                            vam    = float(str(row.get("VAM (m/h)", 0)).replace(",", ".") or 0)
                        except (ValueError, TypeError):
                            avg_hr, grade, vam = 0.0, 0.0, 0.0

                        adj_vam   = int(round(vam * (1 - (grade/100 - 0.1) * 3)))
                        tom_score = round(adj_vam / avg_hr, 1) if avg_hr > 0 else 0.0

                        tom_rows.append({
                            "Climb":          name,
                            "AVG HR (bpm)":   int(avg_hr),
                            "Avg Grade (%)":  grade,
                            "VAM (m/h)":      int(vam),
                            "ADJ VAM":        adj_vam,
                            "TOM LOW SCORE":  tom_score,
                        })

                    tom_df = pd.DataFrame(tom_rows)
                    st.markdown("#### 🔒 Tom Low Secret Data")
                    st.dataframe(tom_df, use_container_width=True, hide_index=True)

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
        # Banda scura a tutta larghezza: il logo DU è bianco, quindi ha
        # bisogno di questo fondo per essere visibile.
        self.set_fill_color(30, 30, 30)
        self.rect(0, 0, 210, 32, 'F')

        text_x = 10
        if os.path.exists(LOGO_DU_PATH):
            self.image(LOGO_DU_PATH, x=10, y=6, h=20)
            text_x = 42   # <-- allarga se il logo è più largo del previsto

        if os.path.exists(SCRITTA_PATH):
            self.image(SCRITTA_PATH, x=150, y=9, w=50)

        self.set_xy(text_x, 6)
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.cell(0, 10, "DU COACHING - Race Analyzer Report", ln=True)
        self.set_xy(text_x, 18)
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(200, 200, 200)
        self.cell(0, 6, "This analyzer is brought to you by Coach Ambro", ln=True)
        self.ln(7)

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

    fig, ax = plt.subplots(figsize=(10, 1.5))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, y_kde.max() * 1.35)

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

    # Titolo dentro gli assi e niente etichetta x, come nei grafici a schermo:
    # su un A4 verticale ogni grafico risparmia così ~15 mm.
    ax.set_ylabel("Probability (%)", fontsize=9)
    ax.text(
        0.01, 0.97, title, transform=ax.transAxes,
        ha="left", va="top", fontsize=10, fontweight="bold", color="black",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
    )
    ax.tick_params(labelsize=8)
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
        pdf.cell(col_width, row_height, str(col).replace(" (km/h)", " km/h")[:14],
                 border=1, fill=True, align='C')
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
if analysis_ready and 'df' in locals() and not df.empty and 'HR Zone' in df.columns and 'athlete_name' in st.session_state:
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

        # --- HR + EFS + Elevation Trend Chart ---
        # Tre bande impilate con asse x condiviso, stessa lettura del
        # grafico live. Le proporzioni ricalcano i pesi della UI
        # (42/28/30). efs_df arriva dalla sezione live, già calcolato.
        _efs_ok = (
            'efs_df' in globals()
            and efs_df is not None
            and "efs_smooth" in efs_df.columns
            and efs_df["efs_smooth"].notna().any()
        )

        _n_bands = 1 + (1 if _efs_ok else 0) + 1   # HR + EFS? + elevazione
        _ratios = [42, 28, 30] if _efs_ok else [60, 40]

        fig, axes = plt.subplots(
            _n_bands, 1, figsize=(10, 6.5), sharex=True,
            gridspec_kw={"height_ratios": _ratios, "hspace": 0.10},
        )

        ax_hr = axes[0]
        ax_hr.plot(df_clean["elapsed_hours"], df_clean["hr_smooth"],
                   color="royalblue", linewidth=0.8, label="Heart Rate")
        try:
            ax_hr.plot(df_clean["elapsed_hours"], reg.predict(X),
                       color="red", linestyle="--", linewidth=1.8, label="HR Trend")
        except Exception:
            pass
        ax_hr.set_ylabel("Heart Rate (bpm)", fontsize=9)
        ax_hr.grid(True, alpha=0.3)
        ax_hr.legend(loc="upper right", fontsize=7, ncol=2)
        ax_hr.tick_params(labelsize=8)
        ax_hr.set_title("Heart Rate, Equivalent Flat Speed & Elevation",
                        fontsize=11, fontweight="bold")

        if _efs_ok:
            ax_efs = axes[1]
            ax_efs.plot(efs_df["elapsed_hours"], efs_df["efs_smooth"],
                        color="seagreen", linewidth=1.0, label="EFS")
            _efs_valid_pdf = efs_df.dropna(subset=["efs_kmh"])
            if len(_efs_valid_pdf) > 2:
                _Xp = _efs_valid_pdf["elapsed_hours"].values.reshape(-1, 1)
                _regp = LinearRegression().fit(_Xp, _efs_valid_pdf["efs_kmh"].values)
                ax_efs.plot(_efs_valid_pdf["elapsed_hours"], _regp.predict(_Xp),
                            color="darkgreen", linestyle="--", linewidth=1.8,
                            label="EFS Trend")
            ax_efs.set_ylabel("EFS (km/h)", fontsize=9)
            ax_efs.grid(True, alpha=0.3)
            ax_efs.legend(loc="upper right", fontsize=7, ncol=2)
            ax_efs.tick_params(labelsize=8)

        ax_ele = axes[-1]
        _ele_x = df["elapsed_sec"] / 3600.0
        _ele_y = df["elevation_m"].rolling(window=20, min_periods=1).mean()
        ax_ele.fill_between(_ele_x, _ele_y, _ele_y.min(),
                            color="gray", alpha=0.35)
        ax_ele.plot(_ele_x, _ele_y, color="dimgray", linewidth=0.9)
        ax_ele.set_ylabel("Elev (m)", fontsize=9)
        ax_ele.set_xlabel("Elapsed Time (hours)", fontsize=9)
        ax_ele.tick_params(labelsize=8)

        plt.tight_layout()
        add_chart_to_pdf(fig, title="Trend Analysis - HR, EFS & Elevation")

        # Stessi due indici della sezione live: DET sulla FC, SPD sulla
        # velocità, ciascuno con il tasso grezzo accanto.
        _hr_rate_pdf = float(reg.coef_[0]) * 3600.0
        pdf.body_text(
            f"DET Index: {det_index_str} ({comment})   |   "
            f"{_hr_rate_pdf:+.3f} bpm per hour"
        )
        if 'spd_index' in globals() and spd_index is not None:
            if spd_index < 0.8:
                _spd_cmt = "Scarso decadimento"
            elif spd_index <= 2.0:
                _spd_cmt = "Decadimento medio"
            else:
                _spd_cmt = "Alto decadimento"
            pdf.body_text(
                f"SPD Index: {spd_index:.2f} ({_spd_cmt})   |   "
                f"{-spd_kmh_per_h:+.3f} km/h per hour"
            )
        if 'efs_totals' in globals() and efs_totals is not None:
            pdf.body_text(
                f"Total EFD: {efs_totals['efd_km']:.2f} km  |  "
                f"Average EFS: {efs_totals['avg_efs_kmh']:.2f} km/h"
            )
        pdf.add_spacer(4)

                # --- Efficiency Factor ---
        # _ef, _ef_x, _ef_rel, _ef_efs, _hr_thr arrivano dalla sezione live.
        # Se l'analisi EF non è stata eseguita (traccia corta, soglia non
        # impostata) il blocco si salta senza rompere il report.
        if ('_ef' in globals() and _ef is not None
                and '_ef_x' in globals() and len(_ef_x) > 3):
            pdf.add_page()

            fig_ефp, ax_ef = plt.subplots(figsize=(10.5, 4))

            # profilo altimetrico di sfondo, schiacciato in basso
            _bgp_x = df["elapsed_sec"] / 3600.0
            _bgp_y = df["elevation_m"].rolling(window=20, min_periods=1).mean()
            ax_bg = ax_ef.twinx()
            ax_bg.fill_between(_bgp_x, _bgp_y, _bgp_y.min(),
                               color="gray", alpha=0.18, zorder=0)
            ax_bg.set_ylim(float(_bgp_y.min()), float(_bgp_y.max()) * 2.6)
            ax_bg.set_yticks([])

            ax_ef.plot(_ef_x, _ef, color="#e07b39", linewidth=2.2,
                       label="EF (km/h a soglia)", zorder=3)
            ax_ef.set_ylabel("EF (km/h per sforzo relativo)", fontsize=9)
            ax_ef.set_xlabel("Elapsed Time (hours)", fontsize=9)
            ax_ef.grid(True, alpha=0.3)
            ax_ef.tick_params(labelsize=8)
            ax_ef.set_zorder(ax_bg.get_zorder() + 1)
            ax_ef.patch.set_visible(False)

            ax_hrp = ax_ef.twinx()
            ax_hrp.plot(_ef_x, _ef_rel * 100.0, color="royalblue",
                        linewidth=1.1, alpha=0.7, label="FC (% soglia)")
            ax_hrp.set_ylabel("FC (% soglia)", fontsize=9)
            ax_hrp.tick_params(labelsize=8)

            # Terzo asse spostato in fuori: senza offset l'etichetta finisce
            # sopra quella della FC e diventano illeggibili entrambe.
            ax_efsp = ax_ef.twinx()
            ax_efsp.spines["right"].set_position(("outward", 42))
            ax_efsp.plot(_ef_x, _ef_efs, color="seagreen",
                         linewidth=1.1, alpha=0.7, label="EFS (km/h)")
            ax_efsp.set_ylabel("EFS (km/h)", fontsize=9)
            ax_efsp.tick_params(labelsize=8)

            _h1, _l1 = ax_ef.get_legend_handles_labels()
            _h2, _l2 = ax_hrp.get_legend_handles_labels()
            _h3, _l3 = ax_efsp.get_legend_handles_labels()
            ax_ef.legend(_h1 + _h2 + _h3, _l1 + _l2 + _l3,
                         loc="upper right", fontsize=7, ncol=3)
            ax_ef.set_title("Efficiency Factor — FC relativa alla soglia",
                            fontsize=11, fontweight="bold")

            plt.tight_layout()
            add_chart_to_pdf(fig_ефp, title="Efficiency Factor")

            _half_p = len(_ef) // 2
            _ef1_p, _ef2_p = _ef[:_half_p].mean(), _ef[_half_p:].mean()
            _ef_delta_p = _ef2_p - _ef1_p
            _ef_cmt_p, _ = ef_delta_verdict(_ef_delta_p)
            pdf.body_text(
                f"EF 1st half: {_ef1_p:.2f}   |   "
                f"EF 2nd half: {_ef2_p:.2f}   |   "
                f"Race average: {_ef.mean():.2f}"
            )
            pdf.body_text(
                f"EF change: {_ef_delta_p:+.2f} ({_ef_cmt_p})"
            )
            pdf.body_text(
                f"EF = EFS / (HR / threshold), threshold {_hr_thr:.0f} bpm. "
                "Equivalent flat speed the athlete holds per unit of relative "
                "effort."
            )
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