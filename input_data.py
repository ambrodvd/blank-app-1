"""1) Upload del file FIT, dati atleta/gara e zone cardiache."""
import gzip
import io
import math
import warnings

import streamlit as st
import fitdecode
import numpy as np
import pandas as pd


# ======================================================================
# HELPERS
# ======================================================================
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
        except Exception:
            raise ValueError("The .gz file is damaged or isn't a real gzip archive.")

    # Header FIT: byte 8-11 = ".FIT". Qualsiasi altro file (anche rinominato
    # .fit, o un .gz con dentro un GPX/CSV) si ferma qui.
    if len(file_bytes) < 14 or file_bytes[8:12] != b".FIT":
        raise ValueError("This is not a FIT file. Upload the .fit (or .fit.gz) exported "
                         "from the watch or from Garmin Connect / Strava / COROS.")

    records = []
    meta = {"file_type": None, "sport": None, "sub_sport": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            with fitdecode.FitReader(io.BytesIO(file_bytes)) as fit:
                for frame in fit:
                    if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                        continue
                    if frame.name == "file_id" and meta["file_type"] is None:
                        meta["file_type"] = _field(frame, "type")
                    elif frame.name in ("sport", "session") and meta["sport"] is None:
                        meta["sport"] = _field(frame, "sport")
                        meta["sub_sport"] = _field(frame, "sub_sport")
                    elif frame.name == "record":
                        row = {f.name: f.value for f in frame.fields
                               if f.name in ["timestamp", "heart_rate", "distance",
                                              "enhanced_altitude", "position_lat", "position_long"]}
                        if row:
                            records.append(row)
    except fitdecode.FitError as e:
        if not records:
            raise ValueError(f"The FIT file is damaged and can't be read ({e}).")
        # file troncato (es. registrazione interrotta): si usa quello che c'è

    if not records:
        raise ValueError("This FIT file has no recorded data points "
                         "(it may be a course, a workout or a settings file).")

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

    return df, int(kilometers), int(total_elevation_gain), meta


def _field(frame, name):
    """Valore di un campo FIT come testo minuscolo (None se manca)."""
    try:
        v = frame.get_value(name)
    except KeyError:
        return None
    return str(v).lower() if v is not None else None


_RUN_SPORTS = {"running", "walking", "hiking", "generic", "mountaineering",
               "training", "transition", "multisport", "1", "0", "11", "17", "16"}


def check_race_df(df, meta):
    """Controlli sul FIT caricato. Ritorna (errori, avvisi): con un errore
    l'analisi non parte, gli avvisi si mostrano e basta."""
    errors, notes = [], []
    ftype = meta.get("file_type")
    if ftype and ftype not in ("activity", "4"):
        errors.append(f"This FIT file is a **{ftype}** file, not a recorded activity. "
                      "Upload the activity of the race.")
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        errors.append("The file has no timestamps, so times and durations can't be computed.")
    hr = df["heart_rate"] if "heart_rate" in df.columns else pd.Series(dtype=float)
    hr_ok = pd.to_numeric(hr, errors="coerce").gt(0)
    if not hr_ok.any():
        errors.append("The file has **no heart-rate data**: the whole analysis is based on HR.")
    elif hr_ok.mean() < 0.5:
        notes.append(f"Heart rate is missing in {round((1 - hr_ok.mean()) * 100)}% of the "
                     "file: zones, DET and EF may be unreliable.")
    if df["distance_km"].max() <= 0:
        errors.append("The file has **no distance data**.")
    dur_min = (df["elapsed_sec"].max() or 0) / 60
    if not errors and (dur_min < 10 or df["distance_km"].max() < 1):
        errors.append(f"The activity is only {dur_min:.0f} min / {df['distance_km'].max():.1f} km: "
                      "too short to be a race.")
    if df["enhanced_altitude"].isna().all():
        notes.append("The file has no altitude data: elevation, climbs and VAM will be empty.")
    sport = meta.get("sport")
    if sport and sport not in _RUN_SPORTS:
        notes.append(f"This activity was recorded as **{sport}**, not running: "
                     "is it the right file?")
    return errors, notes


def check_reopened_race(df, kilometers, info):
    """Il FIT corrisponde alla gara riaperta? Confronta distanza (±3%) e data (±1 giorno).
    Ritorna l'elenco delle differenze (vuoto = tutto ok)."""
    diffs = []
    ref_km = info.get("km")
    try:
        ref_km = float(ref_km) if ref_km is not None else None
    except (TypeError, ValueError):
        ref_km = None
    if ref_km and kilometers and abs(kilometers - ref_km) / ref_km > 0.03:
        diffs.append(f"distance {kilometers} km instead of {ref_km:g} km")
    ref_date = info.get("date")
    if ref_date and "timestamp" in df.columns and df["timestamp"].notna().any():
        try:
            from datetime import date as _date
            d_ref = _date.fromisoformat(str(ref_date)[:10])
            d_fit = pd.to_datetime(df["timestamp"].dropna().iloc[0]).date()
            if abs((d_fit - d_ref).days) > 1:
                diffs.append(f"date {d_fit:%d/%m/%Y} instead of {d_ref:%d/%m/%Y}")
        except (TypeError, ValueError):
            pass
    return diffs


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


def hhmm_to_seconds(hmm):
    """'1:30' -> 5400. None se non parsabile."""
    try:
        h, m = str(hmm).split(":")
        return int(h) * 3600 + int(m) * 60
    except (ValueError, AttributeError):
        return None


def hhmm_to_minutes(hmm):
    """'1:30' -> 90. 0 se non parsabile (celle vuote di tabelle/heatmap)."""
    try:
        h, m = map(int, str(hmm).split(":"))
        return h * 60 + m
    except (ValueError, AttributeError):
        return 0


def parse_time_to_seconds(t):
    """Versione permissiva: accetta H:MM, H:MM:SS, numeri o vuoto. 0 se non parsabile."""
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
    """5430 -> '1:30:30'."""
    sec = int(sec)
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"


def format_hm(sec):
    """5430 -> '1:30' (ore senza zero iniziale, a differenza di seconds_to_hhmm)."""
    sec = int(sec)
    return f"{sec // 3600}:{(sec % 3600) // 60:02d}"


def hhmm_axis_formatter(x, pos):
    """FuncFormatter matplotlib: secondi sull'asse x -> 'hh:mm'."""
    return seconds_to_hhmm(x)


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


# Etichette delle zone FC, nell'ordine Z1 -> Z5 (colonna df["HR Zone"])
HR_ZONE_LABELS = ["Zone 1 // Aerobic Low", "Zone 2 // Aerobic High",
                  "Zone 3 // Aerobic Endurance", "Zone 4 // Sub Threshold",
                  "Zone 5 // Super Threshold"]

# Palette categoriale per i grafici Lap / Climb: 8 tinte in ordine fisso,
# validata per daltonismo (deuter/protan/tritan) su sfondo chiaro e scuro.
CHART_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                 "#e87ba4", "#008300", "#6250d6", "#e34948"]

# Colori dei segmenti / lap / salite in grafici live e PDF
SEGMENT_COLORS = ["royalblue", "tomato", "gold", "mediumseagreen", "orchid",
                  "darkorange", "deepskyblue", "limegreen", "crimson", "slateblue"]


def hr_zone_label(hr, z1, z2, z3, z4):
    """Etichetta della zona FC per un singolo valore di FC."""
    if hr <= z1:
        return HR_ZONE_LABELS[0]
    elif hr <= z2:
        return HR_ZONE_LABELS[1]
    elif hr <= z3:
        return HR_ZONE_LABELS[2]
    elif hr <= z4:
        return HR_ZONE_LABELS[3]
    else:
        return HR_ZONE_LABELS[4]


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "analysis_ready", "athlete_name", "df", "kilometers", "race_name",
    "total_elevation_gain", "uploaded_file", "z1", "z2", "z3", "z4", "z5",
)


def render(ctx):
    st.write("")
    st.markdown("### 🏁 Race File")
    reopen_ph = st.empty()     # avviso "carica il FIT della gara riaperta", sopra l'uploader
    st.markdown("*For large race files, to speed up the analysis, first add the race and cardiac data, and then upload the .fit or .gzip file*")
    uploaded_file = st.file_uploader("Upload a .fit or .fit.gz file", type=["fit", "gz"])

    _reo = st.session_state.get("_reopened")
    if _reo and uploaded_file is None:
        _km = f", {_reo['km']} km" if _reo.get("km") else ""
        reopen_ph.warning(f"↩️ You reopened **{_reo['label']}{_km}**. Upload the .fit / .fit.gz "
                          "file of **this race** below to see its analysis again.")

    if uploaded_file is not None:
        fit_ok = False
        try:
            df, kilometers, total_elevation_gain, fit_meta = parse_fit_file(uploaded_file.getvalue())
        except Exception as e:
            st.error(f"❌ {e}" if isinstance(e, ValueError) else f"❌ Error reading FIT file: {e}")
        else:
            errors, notes = check_race_df(df, fit_meta)
            for msg in errors:
                st.error(f"❌ {msg}")
            for msg in notes:
                st.warning(f"⚠️ {msg}")
            fit_ok = not errors
            if fit_ok and _reo:
                diffs = check_reopened_race(df, kilometers, _reo)
                if diffs:
                    st.error(f"❌ This doesn't look like the FIT of **{_reo['label']}**: "
                             + "; ".join(diffs) + ".")
                    fit_ok = st.checkbox("It's the right file, use it anyway",
                                         key="fit_mismatch_ok")
            if fit_ok:
                st.success("FIT file uploaded!")

        if fit_ok:
            # Save results in session state so the rest of the app can use them
            # without needing to touch the (cached) parsing function again
            st.session_state['fit_df'] = df
            st.session_state['kilometers'] = kilometers
            st.session_state['total_elevation_gain'] = total_elevation_gain
        else:
            # File non valido: niente analisi, e niente dati del FIT precedente
            for k in ("fit_df", "kilometers", "total_elevation_gain"):
                st.session_state.pop(k, None)
            uploaded_file = None
            df = None
    else:
        st.info("👆 Please upload a .fit file to begin.")

    hr_zones_ready = bool(st.session_state.get("hr_zones_confirmed"))
    analysis_ready = (uploaded_file is not None) and hr_zones_ready

    # --- Athlete and race info form ---
    st.divider()
    st.write("")
    with st.form("race_info_form"):
        athlete_name = st.text_input("🏃 Athlete's Name",
                                     value=st.session_state.get('athlete_name',
                                                                st.session_state.get('athlete_name_prefill', '')))
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

    default_zones = {'z1': 140, 'z2': 160, 'z3': 170, 'z4': 180, 'z5': 200}
    for zone, val in default_zones.items():
        if zone not in st.session_state:
            st.session_state[zone] = val

    # Con "Import CSV" come opzione di default, al primo giro nessun widget
    # assegna z1..z5: il ramo Manual Input non viene eseguito e quello CSV
    # definisce i nomi solo dopo l'upload. Senza questo seeding, premere
    # "Submit HR Zones" prima di caricare un file solleva NameError.
    z1, z2, z3, z4, z5 = (st.session_state['z1'], st.session_state['z2'],
                          st.session_state['z3'], st.session_state['z4'],
                          st.session_state['z5'])

    st.subheader("❤️ Athlete Heart Rate Zones")

    # --- Zone dal curriculum ---
    # Se è stato caricato un curriculum con le zone dell'ultima gara, CSV e
    # inserimento manuale spariscono: restano solo il riepilogo e il Submit.
    # La checkbox "edit" riapre l'inserimento se le zone sono cambiate.
    _cv = st.session_state.get("curriculum_in")
    _cv_zones = None
    _cv_race = None
    if _cv and _cv.get("races"):
        # zone della gara riaperta, altrimenti dell'ultima gara del curriculum
        _rid = (st.session_state.get("_reopened") or {}).get("race_id")
        _cv_race = next((r for r in _cv["races"] if r.get("race_id") == _rid), _cv["races"][-1])
        _zz = _cv_race.get("hr_zones") or {}
        if all(_zz.get(k) is not None for k in ('z1', 'z2', 'z3', 'z4', 'z5')):
            _cv_zones = _zz

    use_cv_zones = False
    if _cv_zones is not None:
        _last = _cv_race.get("race", {})
        _which = "reopened race" if st.session_state.get("_reopened") else "last race"
        st.success(
            f"✅ **HR zones found in the curriculum** "
            f"({_which}: {_last.get('name', '?')}, {_last.get('date', '?')})  \n"
            f"🩵 Z1 ≤ {z1} · 💚 Z2 ≤ {z2} · 💛 Z3 ≤ {z3} · 🧡 Z4 ≤ {z4} · ❤️ Z5 ≤ {z5} bpm  \n"
            "Press **Submit HR Zones** to confirm them."
        )
        use_cv_zones = not st.checkbox("✏️ Zones changed since the last race: edit them",
                                       key="cv_zones_edit")

    # --- Input method ---
    if use_cv_zones:
        input_method = None
    else:
        input_method = st.radio("Select input method:", ["Import CSV", "Manual Input"])

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

    # --- Colonna "HR Zone" ---
    # Calcolata qui, una volta sola, con le zone CONFERMATE (session_state,
    # non i widget non ancora salvati): la usano segment analysis, density,
    # heatmap, lap/climb report e PDF.
    if analysis_ready and all(k in st.session_state for k in ['z1', 'z2', 'z3', 'z4', 'z5']):
        _zones = [st.session_state[k] for k in ('z1', 'z2', 'z3', 'z4')]
        df["HR Zone"] = df["heart_rate"].apply(lambda hr: hr_zone_label(hr, *_zones))

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})