"""10) Curriculum dell'atleta: storico delle gare in un unico file JSON.

Il file resta sul PC dell'utente: si scarica dopo l'analisi e si ricarica
la volta successiva per aggiungere la gara nuova. Un file per atleta.

Struttura del file:
    {
      "schema_version": 1,
      "athlete_id":   "a1b2c3d4...",      # fisso dalla creazione del file
      "athlete_name": "Mario Rossi",
      "races": [ {race record}, ... ]     # ordinate per data
    }

Le funzioni load_curriculum / build_race_record / add_race sono a livello
di modulo per poterle riusare nelle prossime feature che leggono il file.
"""
import json
import math
import re
import uuid
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd

from input_data import hhmm_to_minutes


# ======================================================================
# HELPERS
# ======================================================================
# Versione dell'app scritta in ogni gara: va aggiornata quando cambia un
# calcolo (scala SPD, soglie DET/EF, blocchi EF...), così si sa quali gare
# del curriculum sono state calcolate con le regole vecchie.
APP_VERSION = "2026.11.06"
SCHEMA_VERSION = 1


def _clean(obj):
    """Rende serializzabile in JSON: tipi numpy -> Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _num(x, ndigits=2):
    """float arrotondato, None se mancante o non numerico."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return round(f, ndigits) if math.isfinite(f) else None


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_") or "x"


def race_id(race_name, race_date):
    """Chiave della gara: stessa data + stesso nome = stessa gara (sovrascrive)."""
    return f"{race_date}_{_slug(race_name)}"


def new_curriculum(athlete_name):
    return {
        "schema_version": SCHEMA_VERSION,
        "athlete_id": uuid.uuid4().hex[:12],
        "athlete_name": athlete_name,
        "races": [],
    }


def load_curriculum(uploaded_file):
    """Legge e valida un file curriculum caricato. Solleva ValueError se non valido."""
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
    except Exception as e:
        raise ValueError(f"not a valid curriculum file ({e})")
    if not isinstance(data, dict) or "athlete_id" not in data or "races" not in data:
        raise ValueError("this JSON is not a Race Analyzer curriculum")
    if data.get("schema_version", 0) > SCHEMA_VERSION:
        raise ValueError("this curriculum was made by a newer version of the app")
    return data


def add_race(curriculum, record):
    """Aggiunge (o sostituisce, se stesso race_id) una gara. Ritorna (curriculum, replaced)."""
    races = [r for r in curriculum["races"] if r.get("race_id") != record["race_id"]]
    replaced = len(races) != len(curriculum["races"])
    races.append(record)
    races.sort(key=lambda r: (r.get("race", {}).get("date") or ""))
    curriculum["races"] = races
    return curriculum, replaced


def _table_records(table):
    """DataFrame -> lista di dict (tabelle Lap / Climb così come mostrate in app)."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return []
    return table.reset_index(drop=True).to_dict(orient="records")


def _time_in_zone_records(combined_df):
    """Tabella tempo in zona (celle 'H:MM') -> righe zona/colonna/minuti."""
    if combined_df is None or combined_df.empty:
        return []
    rows = []
    for zone, row in combined_df.iterrows():
        for col, val in row.items():
            rows.append({"zone": zone, "segment": col, "time_hmm": val,
                         "minutes": hhmm_to_minutes(val)})
    return rows


def build_race_record(ctx):
    """Raccoglie in un dict tutti i dati della gara analizzata."""
    ss = st.session_state
    df = ctx.get("df")
    race_name = ss.get("race_name", "")
    race_date = ss.get("race_date")
    race_date = race_date.isoformat() if race_date is not None else None

    n_seg = ss.get("num_segments", 0) or 0
    segments = [
        {"start": ss.get(f"segment{i}_start"), "end": ss.get(f"segment{i}_end")}
        for i in range(1, n_seg + 1)
        if ss.get(f"segment{i}_start") is not None
    ]

    efs_totals = ctx.get("efs_totals") or {}
    reg = ctx.get("reg")
    hr_drift = ctx.get("_hr_bpm_per_h")
    if hr_drift is None and reg is not None:
        hr_drift = float(reg.coef_[0]) * 3600.0

    climbs = ss.get("climb_zone_df") if ss.get("do_climb_analysis") else None
    laps = ss.get("lap_zone_df") if ss.get("do_lap_analysis") else None

    record = {
        "race_id": race_id(race_name, race_date),
        "analysis_date": datetime.now().isoformat(timespec="seconds"),
        "app_version": APP_VERSION,
        "race": {
            "name": race_name,
            "date": race_date,
            "distance_km": _num(ctx.get("kilometers"), 1),
            "elevation_gain_m": _num(ctx.get("total_elevation_gain"), 0),
            "final_time": ctx.get("final_time_str"),
        },
        "hr_zones": {k: (int(ss[k]) if ss.get(k) is not None else None)
                     for k in ("z1", "z2", "z3", "z4", "z5")},
        "time_segments": segments,
        "heart_rate": {
            "overall_avg": _num(ctx.get("overall_avg"), 1),
            "first_half_avg": _num(ctx.get("first_half_avg"), 1),
            "second_half_avg": _num(ctx.get("second_half_avg"), 1),
            "percent_diff": _num(ctx.get("percent_diff"), 2),
            "max_hr": _num(df["heart_rate"].max(), 0) if df is not None else None,
        },
        "det": {
            "index": _num(ctx.get("det_index_str"), 1),
            "verdict": ctx.get("comment"),
            "hr_drift_bpm_per_h": _num(hr_drift, 2),
        },
        "spd": {
            "index": _num(ctx.get("spd_index"), 2),
            "verdict": ctx.get("spd_comment"),
            "speed_decay_kmh_per_h": _num(ctx.get("spd_kmh_per_h"), 3),
        },
        "efs": {
            "avg_efs_kmh": _num(efs_totals.get("avg_efs_kmh"), 2),
            "total_efd_km": _num(efs_totals.get("efd_km"), 2),
        },
        "ef": {
            "first_half": _num(ctx.get("_ef_1"), 2),
            "second_half": _num(ctx.get("_ef_2"), 2),
            "race_mean": _num(np.nanmean(ctx["_ef"]), 2) if ctx.get("_ef") is not None and len(ctx["_ef"]) else None,
            "delta": _num(ctx.get("_ef_delta"), 2),
            "verdict": ctx.get("_ef_cmt"),
            "block_min": ctx.get("EF_BLOCK_MIN"),
        },
        "time_in_zone": _time_in_zone_records(ctx.get("combined_df")),
        "climbs": _table_records(climbs),
        "laps": _table_records(laps),
        "coach_comment": ss.get("comment", ""),
    }
    return _clean(record)


# ======================================================================
# RENDER — caricamento in cima alla pagina
# ======================================================================
def render_loader(ctx):
    """Primo blocco della pagina: carica il curriculum (facoltativo) e
    precompila nome atleta e zone FC dall'ultima gara salvata.
    La precompilazione avviene UNA volta per file caricato: le modifiche
    fatte dopo a mano nel form non vengono sovrascritte ai rerun."""
    ss = st.session_state
    st.markdown("### 📚 Athlete Curriculum")
    st.caption("Already analysed races for this athlete? Upload the curriculum file: "
               "name and HR zones are filled in for you. New athlete? Skip this.")
    uploaded = st.file_uploader("Upload the athlete curriculum (.json)", type=["json"],
                                key="curriculum_upload")

    if uploaded is None:
        # file rimosso: dimentica il curriculum (nome e zone restano compilati)
        ss.pop("curriculum_in", None)
        ss.pop("curriculum_sig", None)
        return

    sig = (uploaded.name, uploaded.size)
    if ss.get("curriculum_sig") != sig:
        try:
            cur = load_curriculum(uploaded)
        except ValueError as e:
            ss.pop("curriculum_in", None)
            st.error(f"❌ Curriculum not loaded: {e}")
            return
        ss["curriculum_in"] = cur
        ss["curriculum_sig"] = sig
        ss.pop("curriculum_out", None)
        # zone precompilate = da ricontrollare: il gate richiede di nuovo Submit
        ss["hr_zones_confirmed"] = False

        ss["athlete_name"] = cur.get("athlete_name", "")
        last = cur["races"][-1] if cur.get("races") else None
        zones = (last or {}).get("hr_zones") or {}
        if all(zones.get(k) is not None for k in ("z1", "z2", "z3", "z4", "z5")):
            for k in ("z1", "z2", "z3", "z4", "z5"):
                ss[k] = int(zones[k])

    cur = ss.get("curriculum_in")
    if cur is not None:
        last = cur["races"][-1] if cur.get("races") else None
        last_txt = (f" Last race: {last['race'].get('name')} ({last['race'].get('date')})."
                    if last else "")
        st.success(
            f"✅ Curriculum properly uploaded: **{cur.get('athlete_name')}**, "
            f"{len(cur.get('races', []))} race(s).{last_txt} "
            "Athlete name and HR zones have been filled in — check them and press "
            "**Submit HR Zones** to confirm."
        )
    st.markdown("---")


# ======================================================================
# RENDER — salvataggio in fondo alla pagina
# ======================================================================
def render(ctx):
    df = ctx.get("df")
    ss = st.session_state
    if not (ctx.get("analysis_ready") and df is not None and not df.empty
            and "athlete_name" in ss):
        return

    st.markdown("---")
    st.subheader("📚 Athlete Curriculum")
    st.caption(
        "Add this race to the athlete's history and download the file. Next time, upload "
        "it at the top of the page before starting. The file stays on your computer."
    )

    athlete_name = ss.get("athlete_name", "").strip()
    curriculum = ss.get("curriculum_in")
    if curriculum is not None:
        curriculum = json.loads(json.dumps(curriculum))   # copia: il caricato resta intatto
        if _slug(curriculum.get("athlete_name")) != _slug(athlete_name):
            st.error(
                f"❌ This curriculum belongs to **{curriculum.get('athlete_name')}**, "
                f"but the athlete in this analysis is **{athlete_name}**. "
                "Fix the athlete name in the form above or upload the right file at the top."
            )
            return
    else:
        st.info("No curriculum uploaded at the top: a new one will be created for this athlete.")

    if st.button("➕ Add this race to the curriculum", key="curriculum_add"):
        if curriculum is None:
            curriculum = new_curriculum(athlete_name)
        record = build_race_record(ctx)
        curriculum, replaced = add_race(curriculum, record)
        ss["curriculum_out"] = curriculum
        if replaced:
            st.info("🔁 This race was already in the curriculum: it has been updated.")
        else:
            st.success("✅ Race added.")

    out = ss.get("curriculum_out")
    if out is not None and _slug(out.get("athlete_name")) == _slug(athlete_name):
        races = pd.DataFrame([{
            "Date": r["race"].get("date"),
            "Race": r["race"].get("name"),
            "Km": r["race"].get("distance_km"),
            "D+ (m)": r["race"].get("elevation_gain_m"),
            "Time": r["race"].get("final_time"),
            "Avg HR": r["heart_rate"].get("overall_avg"),
            "DET": r["det"].get("index"),
            "SPD": r["spd"].get("index"),
            "Δ EF": r["ef"].get("delta"),
        } for r in out["races"]])
        st.dataframe(races, hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Download curriculum",
            data=json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8"),
            file_name=f"{_slug(athlete_name)}_curriculum.json",
            mime="application/json",
            key="curriculum_download",
        )