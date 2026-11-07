"""10) Curriculum dell'atleta: storico delle gare in un unico file JSON.

Il file resta sul PC dell'utente: si scarica dopo l'analisi e si ricarica
la volta successiva per aggiungere la gara nuova. Un file per atleta.

Struttura del file:
    {
      "schema_version": 2,
      "athlete_id":   "a1b2c3d4...",      # fisso dalla creazione del file
      "athlete_name": "Mario Rossi",
      "races": [ {race record}, ... ]     # ordinate per data
    }

Versioni:
    1  prima versione
    2  ogni salita ha anche "Start" (h:mm) e "Position (%)" = inizio salita
       in % del tempo gara. I file v1 si caricano lo stesso: le salite
       vecchie mostrano "—" per la posizione.
    3  ogni gara ha anche "settings": le impostazioni dell'analisi (segmenti,
       lap, salite, tabelle modificate dal coach, colonne visibili...).
       Servono per riaprire l'analisi caricando di nuovo il FIT. Le gare
       salvate prima si possono confrontare ma non riaprire per intero.

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

from input_data import hhmm_to_minutes, hhmm_to_seconds, seconds_to_hhmm


# ======================================================================
# HELPERS
# ======================================================================
# Versione dell'app scritta in ogni gara: va aggiornata quando cambia un
# calcolo (scala SPD, soglie DET/EF, blocchi EF...), così si sa quali gare
# del curriculum sono state calcolate con le regole vecchie.
APP_VERSION = "2026.11.06"
SCHEMA_VERSION = 3

# Chiavi di session_state salvate/ripristinate per riaprire un'analisi.
# Widget (checkbox, radio, number_input): ripristinati prima che vengano disegnati.
_SETTING_KEYS = [
    "num_segments", "do_lap_checkbox", "do_climb_checkbox",
    "lap_data_insert_selector", "climb_data_insert_selector",
    "climb_min_gain", "climb_min_grade", "climb_smooth",
    "climb_min_len", "climb_merge_gap", "climb_merge_dip",
]
_SETTING_PREFIXES = ("segment", "lap_show_", "climb_show_")
_SETTING_TABLES = ["manual_lap_table", "manual_climb_table", "editable_climb_table"]


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


def races_table(curriculum):
    """Riepilogo di una riga per gara (in cima alla pagina e in fondo)."""
    def g(r, *p):
        for k in p:
            r = r.get(k) if isinstance(r, dict) else None
        return r
    return pd.DataFrame([{
        "Date": g(r, "race", "date"),
        "Race": g(r, "race", "name"),
        "Km": g(r, "race", "distance_km"),
        "D+ (m)": g(r, "race", "elevation_gain_m"),
        "Time": g(r, "race", "final_time"),
        "Avg HR": g(r, "heart_rate", "overall_avg"),
        "DET": g(r, "det", "index"),
        "SPD": g(r, "spd", "index"),
        "Δ EF": g(r, "ef", "delta"),
    } for r in curriculum.get("races", [])])


def _table_records(table):
    """DataFrame -> lista di dict (tabelle Lap / Climb così come mostrate in app)."""
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return []
    return table.reset_index(drop=True).to_dict(orient="records")


def climb_positions(climb_data, df):
    """{nome salita: (inizio in secondi, posizione in % del tempo gara)}.
    Usa start_idx se c'è (detector automatico), altrimenti start_time h:mm."""
    out = {}
    if not climb_data or df is None or df.empty:
        return out
    total = float(df["elapsed_sec"].max())
    for i, c in enumerate(climb_data):
        name = c.get("name", f"Climb {i + 1}")
        start = None
        if c.get("start_idx") is not None and c["start_idx"] in df.index:
            start = float(df.loc[c["start_idx"], "elapsed_sec"])
        if start is None:
            start = hhmm_to_seconds(c.get("start_time"))
        if start is None or total <= 0:
            continue
        out[name] = (start, round(start / total * 100.0))
    return out


def _climb_records(climb_df, climb_data, df):
    """Tabella salite + inizio e posizione nella gara (schema v2)."""
    rows = _table_records(climb_df)
    pos = climb_positions(climb_data, df)
    for r in rows:
        start, pct = pos.get(r.get("Climb Name"), (None, None))
        r["Start"] = seconds_to_hhmm(start) if start is not None else None
        r["Position (%)"] = pct
    return rows


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


def _settings_snapshot():
    """Impostazioni dell'analisi corrente, per poterla riaprire (schema v3)."""
    ss = st.session_state
    snap = {"keys": {}, "tables": {}}
    for k in list(ss.keys()):
        if k in _SETTING_KEYS or (k.startswith(_SETTING_PREFIXES) and not k.endswith("_pdf")):
            v = ss[k]
            if isinstance(v, (bool, int, float, str)) or v is None:
                snap["keys"][k] = v
    for k in _SETTING_TABLES:
        t = ss.get(k)
        if isinstance(t, pd.DataFrame):
            snap["tables"][k] = t.astype(object).where(t.notna(), None).to_dict(orient="records")
    snap["lap_data"] = ss.get("lap_data") or []
    snap["climb_data"] = ss.get("climb_data") or []
    sig = ss.get("_climb_detect_sig")
    snap["climb_detect_sig"] = list(sig) if sig is not None else None
    return snap


def apply_settings(record):
    """Riapre una gara del curriculum: dati gara, zone, impostazioni, commento.
    Da chiamare PRIMA che vengano disegnati i widget (in cima alla pagina).
    Ritorna False se la gara non ha impostazioni (salvata prima della v3)."""
    ss = st.session_state
    race = record.get("race") or {}
    ss["athlete_name"] = ss.get("athlete_name_prefill") or ss.get("athlete_name", "")
    ss["race_name"] = race.get("name", "")
    try:
        ss["race_date"] = datetime.fromisoformat(race.get("date")).date()
    except (TypeError, ValueError):
        pass
    zones = record.get("hr_zones") or {}
    if all(zones.get(k) is not None for k in ("z1", "z2", "z3", "z4", "z5")):
        for k in ("z1", "z2", "z3", "z4", "z5"):
            ss[k] = int(zones[k])
        ss["hr_zones_confirmed"] = True
    ss["comment"] = record.get("coach_comment", "") or ""
    snap = record.get("settings")
    if not snap:
        return _rebuild_settings(record)
    keys = dict(snap.get("keys") or {})
    keys["do_lap_checkbox"] = bool(keys.get("do_lap_checkbox", False))
    keys["do_climb_checkbox"] = bool(keys.get("do_climb_checkbox", False))
    ss["do_lap_analysis"] = keys["do_lap_checkbox"]
    ss["do_climb_analysis"] = keys["do_climb_checkbox"]
    _set_widgets(keys)
    for k, rows in (snap.get("tables") or {}).items():
        ss[k] = pd.DataFrame(rows)
    ss["lap_data"] = snap.get("lap_data") or []
    ss["climb_data"] = snap.get("climb_data") or []
    if snap.get("climb_detect_sig") is not None:
        ss["_climb_detect_sig"] = tuple(snap["climb_detect_sig"])
    return True


def _set_widgets(values, keep=False):
    """Valori dei widget (checkbox, radio, parametri) di una gara riaperta.
    Si scrivono subito e restano "in attesa" in _pending_widgets: apply_pending()
    li riscrive a ogni run finché l'analisi non è stata disegnata con il FIT.
    Serve perché checkbox e tabelle Lap/Climb compaiono solo dopo il caricamento
    del FIT: un valore scritto quando il widget non è a schermo Streamlit lo
    scarta, e il browser rimanderebbe quello vecchio."""
    ss = st.session_state
    pending = dict(ss.get("_pending_widgets") or {}) if keep else {}
    pending.update(values)
    ss["_pending_widgets"] = pending
    # i data editor ripartono puliti: niente modifiche vecchie riapplicate
    ss["_climb_editor_ver"] = ss.get("_climb_editor_ver", 0) + 1
    apply_pending()


def apply_pending():
    """Da chiamare in cima a ogni run, prima dei widget delle sezioni."""
    ss = st.session_state
    for k, v in (ss.get("_pending_widgets") or {}).items():
        ss[k] = v


def clear_pending(analysis_ready):
    """In fondo al run: se l'analisi è stata disegnata con il FIT, i widget
    hanno preso i loro valori e da qui in poi comanda l'utente."""
    if analysis_ready:
        st.session_state.pop("_pending_widgets", None)


def _rebuild_settings(record):
    """Gare salvate prima della v3: ricostruisce segmenti, salite e lap dai
    risultati salvati. Ritorna True se ha ricostruito tutto, False se qualcosa
    manca (es. salite salvate prima della v2, senza ora di inizio)."""
    ss = st.session_state
    complete = True
    ss["_pending_widgets"] = {}

    segs = record.get("time_segments") or []
    if segs:
        ss["num_segments"] = len(segs)
        for i, sgm in enumerate(segs, start=1):
            ss[f"segment{i}_start"] = sgm.get("start")
            ss[f"segment{i}_end"] = sgm.get("end")

    climbs = []
    for c in record.get("climbs") or []:
        start = hhmm_to_seconds(c.get("Start"))
        dur = _hms_seconds(c.get("Duration"))
        if start is None or dur is None:
            complete = False
            climbs = []
            break
        climbs.append({"name": c.get("Climb Name"), "start_time": seconds_to_hhmm(start),
                       "end_time": seconds_to_hhmm(start + dur)})
    if climbs:
        ss["climb_data"] = climbs
        ss["do_climb_analysis"] = True
        _set_widgets({"do_climb_checkbox": True}, keep=True)

    # Lap: uno dopo l'altro dall'inizio della gara (vale per detector e slicer;
    # per lap manuali con pause fra un lap e l'altro va controllato)
    laps, t = [], 0
    for lp in record.get("laps") or []:
        dur = _hms_seconds(lp.get("Duration"))
        if dur is None:
            complete = False
            laps = []
            break
        laps.append({"name": lp.get("Lap Name"), "start_time": seconds_to_hhmm(t),
                     "end_time": seconds_to_hhmm(t + dur)})
        t += dur
    if laps:
        ss["lap_data"] = laps
        ss["do_lap_analysis"] = True
        _set_widgets({"do_lap_checkbox": True}, keep=True)
    return complete


def _hms_seconds(t):
    """'0:35:00' / '1:30' → secondi, None se non leggibile."""
    try:
        p = [int(x) for x in str(t).split(":")]
    except ValueError:
        return None
    return p[0] * 3600 + p[1] * 60 + p[2] if len(p) == 3 else (p[0] * 3600 + p[1] * 60 if len(p) == 2 else None)


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
        "climbs": _climb_records(climbs, ss.get("climb_data"), df),
        "laps": _table_records(laps),
        "coach_comment": ss.get("comment", ""),
        "settings": _settings_snapshot(),
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
    st.divider()
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

        # Il nome precompila soltanto il form: athlete_name vero e proprio si
        # scrive solo con "Submit Info", insieme a gara e data (le sezioni
        # successive danno per scontato che i tre arrivino insieme).
        ss.pop("athlete_name", None)
        ss["athlete_name_prefill"] = cur.get("athlete_name", "")
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
            "Athlete name and HR zones have been filled in: complete the race info and "
            "press **Submit Info**, then **Submit HR Zones** to confirm."
        )
    # la linea di chiusura la disegna comparison.render_top (elenco gare + scelta)


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
        st.dataframe(races_table(out), hide_index=True, use_container_width=True)
        st.download_button(
            "⬇️ Download curriculum",
            data=json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False).encode("utf-8"),
            file_name=f"{_slug(athlete_name)}_curriculum.json",
            mime="application/json",
            key="curriculum_download",
        )