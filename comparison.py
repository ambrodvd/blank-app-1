"""11) Confronto con le gare del curriculum.

In cima alla pagina (render_top): elenco delle gare del curriculum e scelta
delle gare da confrontare. Quella scelta diventa il default di ogni sezione.

Ogni sezione dell'analisi chiama la sua funzione render_* qui sotto, che
disegna un expander con:
  - un multiselect delle gare (precompilato con la scelta in cima, modificabile);
  - un avviso se distanza o D+ della gara scelta differiscono oltre ±20%;
  - la tabella di confronto, con Δ = questa gara − gara selezionata
    (sempre contro OGNI gara scelta, mai contro una media);
  - una checkbox "Add this comparison to the PDF": se spuntata, il
    confronto finisce in fondo al PDF (session_state["cmp_pdf"]).

Se nessun curriculum è caricato in cima alla pagina, non viene mostrato nulla.
Questo modulo non importa nessun file di sezione (niente import circolari):
i valori della gara corrente arrivano come argomenti.
"""
import hashlib
import math

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from climbs import CLIMB_AUTO_NAME_RE
from curriculum import _slug, climb_positions, race_id, races_table
from input_data import CHART_PALETTE, HR_ZONE_LABELS, hhmm_to_minutes


# ======================================================================
# HELPERS
# ======================================================================
TOLERANCE = 0.20          # ±20% su distanza e D+ → avviso di confrontabilità
GRADE_WINDOW = 2          # salite simili: pendenza ±2 punti percentuali
LENGTH_WINDOW = 0.40      # salite simili: lunghezza ±40%
DELTA_CAPTION = "Δ = this race − selected race."
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
SECTIONS = ["total", "detspd", "ef", "tiz", "climbs", "laps"]


def _f(x):
    """Numero da valore di tabella ('9.1', '43%', '-', None...) o None."""
    if x is None:
        return None
    if isinstance(x, (int, float, np.number)):
        x = float(x)
        return x if math.isfinite(x) else None
    try:
        return float(str(x).strip().rstrip("%").replace(",", "."))
    except ValueError:
        return None


def _get(rec, *path):
    """rec['a']['b'] senza KeyError: None se manca un pezzo."""
    for p in path:
        if not isinstance(rec, dict):
            return None
        rec = rec.get(p)
    return rec


def _num(v, nd=2, signed=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def _cell(past, cur, nd=2, signed=False):
    """'9.45 (Δ +0.42)': valore della gara selezionata e scarto della gara corrente."""
    if past is None:
        return "—"
    txt = _num(past, nd, signed)
    if cur is None:
        return txt
    d = round(cur - past, nd) + 0.0      # + 0.0: niente "-0.0"
    return f"{txt} (Δ {d:+.{nd}f})"


def _hmm(minutes):
    if minutes is None:
        return "—"
    m = int(round(minutes))
    return f"{m // 60}:{m % 60:02d}"


def duration_str(df):
    """Tempo totale h:mm:ss dalla colonna elapsed_sec (come il riepilogo gara)."""
    if df is None or df.empty:
        return "—"
    s = float(df["elapsed_sec"].iloc[-1])
    return f"{int(s // 3600)}:{int((s % 3600) // 60):02d}:{int(s % 60):02d}"


def tom_low_score(vam, grade, avg_hr):
    """(ADJ VAM, TOM LOW SCORE): VAM corretta per la pendenza, divisa per la FC.
    Usata da Tom Low Secret Data in lap_climb_report.py."""
    if vam is None or grade is None:
        return None, None
    adj = int(round(vam * (1 - (grade / 100 - 0.1) * 3)))
    score = round(adj / avg_hr, 1) if avg_hr else 0.0
    return adj, score


def dominant_zone(pcts):
    """['43%', '57%', ...] → 'Z2 · 57%'."""
    vals = [_f(p) for p in pcts]
    if not any(v is not None for v in vals):
        return "—"
    i = int(np.nanargmax([v if v is not None else -1 for v in vals]))
    return f"{ZONES[i]} · {vals[i]:.0f}%"


def _past_races():
    """Gare del curriculum caricato, esclusa quella che si sta analizzando."""
    ss = st.session_state
    cur = ss.get("curriculum_in")
    if not cur or not cur.get("races"):
        return []
    this_id = None
    if ss.get("race_name") and ss.get("race_date") is not None:
        this_id = race_id(ss["race_name"], ss["race_date"].isoformat())
    return [r for r in cur["races"] if r.get("race_id") != this_id]


def _label(r):
    return f"{_get(r, 'race', 'date') or '?'} · {_get(r, 'race', 'name') or '?'}"


def _short(r):
    """'Trail del Monte (2025)': nome gara + anno, senza la data completa."""
    d = _get(r, "race", "date") or ""
    return f"{_get(r, 'race', 'name') or '?'} ({d[:4]})" if d else (_get(r, "race", "name") or "?")


def _comparability_warnings(races):
    """Avviso per ogni gara con distanza o D+ oltre ±20% rispetto a questa."""
    ss = st.session_state
    cur_km, cur_dp = _f(ss.get("kilometers")), _f(ss.get("total_elevation_gain"))
    for r in races:
        msgs = []
        for label, cur, past in (("distance", cur_km, _f(_get(r, "race", "distance_km"))),
                                 ("D+", cur_dp, _f(_get(r, "race", "elevation_gain_m")))):
            if cur and past is not None and abs(past - cur) / cur > TOLERANCE:
                msgs.append(f"{label} {(past - cur) / cur * 100:+.0f}%")
        if msgs:
            st.warning(f"⚠️ **{_label(r)}**: {', '.join(msgs)} vs this race "
                       f"(beyond ±{TOLERANCE * 100:.0f}%). Compare with care.")


def _pick(section):
    """Multiselect della sezione, precompilato con la scelta fatta in cima."""
    ss = st.session_state
    races = _past_races()
    by_id = {r["race_id"]: r for r in races}
    key = f"cmp_{section}_races"
    start = ss[key] if key in ss else ss.get("cmp_default_races", [])
    ss[key] = [i for i in start if i in by_id]          # solo gare valide
    chosen = st.multiselect(
        "Races to compare", list(by_id), format_func=lambda i: _label(by_id[i]),
        key=key, placeholder="Choose one or more races from the curriculum",
    )
    races = [by_id[i] for i in chosen]
    _comparability_warnings(races)
    return races


def _pdf_toggle(section, title, tables, chart=None):
    """Checkbox "Add to PDF": salva tabelle (e grafico) per il fondo del PDF."""
    if st.checkbox("📄 Add this comparison to the PDF", key=f"cmp_{section}_pdf"):
        st.session_state.setdefault("cmp_pdf", {})[section] = {
            "title": title, "tables": tables, "chart": chart,
        }


def _this_row(**cols):
    return {"Race": "▶ This race", "Date": _date_this(), **cols}


def _date_this():
    d = st.session_state.get("race_date")
    return d.isoformat() if d is not None else "—"


def _row(r, **cols):
    return {"Race": _get(r, "race", "name") or "?", "Date": _get(r, "race", "date") or "?", **cols}


def _show(table):
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption(DELTA_CAPTION)


def _checkbox_row(label, items, key_prefix):
    """Riga di checkbox (tutte spuntate di default). Ritorna gli item spuntati."""
    st.markdown(f"**{label}**")
    chosen = []
    for start in range(0, len(items), 6):
        chunk = items[start:start + 6]
        for col, (key, text, item) in zip(st.columns(6), chunk):
            if col.checkbox(text, value=True, key=f"{key_prefix}_{key}"):
                chosen.append(item)
    return chosen


# ======================================================================
# IN CIMA ALLA PAGINA: gare del curriculum + scelta di default
# ======================================================================
def render_top():
    ss = st.session_state
    cur = ss.get("curriculum_in")
    if cur and cur.get("races"):
        st.markdown("**Races in the curriculum**")
        st.dataframe(races_table(cur), hide_index=True, use_container_width=True)
        by_id = {r["race_id"]: r for r in cur["races"]}
        sel = st.multiselect(
            "Races to compare with this one (default for every comparison below — "
            "each section can still change it)",
            list(by_id), format_func=lambda i: _label(by_id[i]), key="cmp_default_races",
            placeholder="Choose one or more races",
        )
        # Scelta cambiata in cima → diventa la scelta di ogni sezione
        if ss.get("_cmp_default_prev") != sel:
            ss["_cmp_default_prev"] = list(sel)
            for s in SECTIONS:
                ss[f"cmp_{s}_races"] = list(sel)
    st.markdown("---")


# ======================================================================
# 1) TOTAL DATA
# ======================================================================
def render_total(df, overall_avg, percent_diff, avg_efs_fn):
    """Avg EFS, FC media, % differenza fra le metà, D+/km. Distanza, D+ e
    tempo finale solo come contesto. avg_efs_fn() calcola l'EFS medio della
    gara corrente solo se serve (calcolo non banale)."""
    if not _past_races():
        return
    with st.expander("🔍 Compare total data with past races"):
        races = _pick("total")
        if not races:
            return
        ss = st.session_state
        km, dp = _f(ss.get("kilometers")), _f(ss.get("total_elevation_gain"))
        cur = {
            "efs": _f(avg_efs_fn()),
            "hr": _f(overall_avg),
            "pdiff": _f(percent_diff),
            "dpk": dp / km if km and dp is not None else None,
        }
        rows = [_this_row(**{
            "Distance (km)": _num(km, 1), "D+ (m)": _num(dp, 0),
            "Final time": duration_str(df),
            "Avg EFS (km/h)": _num(cur["efs"]), "Avg HR (bpm)": _num(cur["hr"], 0),
            "HR 2nd vs 1st half (%)": _num(cur["pdiff"], 1, signed=True),
            "D+/km (m/km)": _num(cur["dpk"], 0),
        })]
        for r in races:
            pkm, pdp = _f(_get(r, "race", "distance_km")), _f(_get(r, "race", "elevation_gain_m"))
            rows.append(_row(r, **{
                "Distance (km)": _num(pkm, 1), "D+ (m)": _num(pdp, 0),
                "Final time": _get(r, "race", "final_time") or "—",
                "Avg EFS (km/h)": _cell(_f(_get(r, "efs", "avg_efs_kmh")), cur["efs"]),
                "Avg HR (bpm)": _cell(_f(_get(r, "heart_rate", "overall_avg")), cur["hr"], 0),
                "HR 2nd vs 1st half (%)": _cell(_f(_get(r, "heart_rate", "percent_diff")),
                                                cur["pdiff"], 1, signed=True),
                "D+/km (m/km)": _cell(pdp / pkm if pkm and pdp is not None else None, cur["dpk"], 0),
            }))
        table = pd.DataFrame(rows)
        _show(table)
        st.caption("Distance, D+ and final time are context: they describe the course, "
                   "not the performance.")
        _pdf_toggle("total", "Total data", [("", table)])


# ======================================================================
# 2) DET + SPD
# ======================================================================
def render_det_spd(df, det_index, hr_drift, det_verdict, spd_index, spd_kmh_per_h, spd_verdict):
    """Speed decay mostrato come nel badge: negativo = rallenta."""
    if not _past_races():
        return
    # spazio fra i badge DET/SPD e l'expander (altrimenti si toccano)
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    with st.expander("🔍 Compare DET & SPD with past races"):
        races = _pick("detspd")
        if not races:
            return
        cur_det, cur_drift = _f(det_index), _f(hr_drift)
        cur_spd = _f(spd_index)
        cur_dec = -_f(spd_kmh_per_h) if _f(spd_kmh_per_h) is not None else None
        rows = [_this_row(**{
            "Duration": duration_str(df),
            "DET index": _num(cur_det, 1),
            "HR drift (bpm/h)": _num(cur_drift, 2, signed=True),
            "DET verdict": det_verdict or "—",
            "SPD index": _num(cur_spd),
            "Speed decay (km/h per h)": _num(cur_dec, 3, signed=True),
            "SPD verdict": spd_verdict or "—",
        })]
        for r in races:
            past_dec = _f(_get(r, "spd", "speed_decay_kmh_per_h"))
            rows.append(_row(r, **{
                "Duration": _get(r, "race", "final_time") or "—",
                "DET index": _cell(_f(_get(r, "det", "index")), cur_det, 1),
                "HR drift (bpm/h)": _cell(_f(_get(r, "det", "hr_drift_bpm_per_h")),
                                          cur_drift, 2, signed=True),
                "DET verdict": _get(r, "det", "verdict") or "—",
                "SPD index": _cell(_f(_get(r, "spd", "index")), cur_spd),
                "Speed decay (km/h per h)": _cell(-past_dec if past_dec is not None else None,
                                                  cur_dec, 3, signed=True),
                "SPD verdict": _get(r, "spd", "verdict") or "—",
            }))
        table = pd.DataFrame(rows)
        _show(table)
        st.caption("A long race has more room to drift than a short one: read DET "
                   "together with the duration.")

        # --- Trend DET: gare scelte + questa, in ordine di data ---
        pts = [(_get(r, "race", "date"), _get(r, "race", "name"),
                _f(_get(r, "det", "index")), False) for r in races]
        pts.append((_date_this(), "This race", cur_det, True))
        pts = sorted([p for p in pts if p[0] and p[0] != "—" and p[2] is not None],
                     key=lambda p: p[0])
        if len(pts) >= 2:
            st.plotly_chart(det_trend_figure(pts), use_container_width=True)
        _pdf_toggle("detspd", "DET & SPD", [("", table)], chart=("det_trend", pts))


def det_trend_figure(pts):
    """DET nel tempo: linea grigia, questa gara evidenziata, soglie 4 e 10."""
    x = pd.to_datetime([p[0] for p in pts])
    y = [p[2] for p in pts]
    names = [p[1] for p in pts]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="lines+markers", showlegend=False,
        line=dict(color="rgba(140,140,140,0.8)", width=2),
        marker=dict(size=9, color="rgba(140,140,140,1)"),
        customdata=names, hovertemplate="%{customdata}<br>%{x|%d/%m/%Y}: DET %{y:.1f}<extra></extra>",
    ))
    cur = [i for i, p in enumerate(pts) if p[3]]
    if cur:
        i = cur[0]
        fig.add_trace(go.Scatter(
            x=[x[i]], y=[y[i]], mode="markers+text", showlegend=False,
            marker=dict(size=14, color=CHART_PALETTE[0], line=dict(width=2, color="white")),
            text=["This race"], textposition="top center",
            hovertemplate="This race<br>%{x|%d/%m/%Y}: DET %{y:.1f}<extra></extra>",
        ))
    for thr, lab in ((4, "4 · scarso"), (10, "10 · alto")):
        fig.add_hline(y=thr, line=dict(color="rgba(128,128,128,0.6)", dash="dot", width=1),
                      annotation_text=lab, annotation_position="right",
                      annotation_font=dict(size=10, color="rgba(110,110,110,1)"))
    fig.update_layout(
        title="DET over time", height=320, margin=dict(t=50, b=40, l=50, r=80),
        xaxis=dict(title=None, showgrid=False),
        yaxis=dict(title="DET index", rangemode="tozero", gridcolor="rgba(128,128,128,0.18)"),
    )
    return fig


# ======================================================================
# 4) EF
# ======================================================================
def render_ef(df, ef_1, ef_2, ef_mean, ef_delta, verdict, block_min):
    if not _past_races():
        return
    with st.expander("🔍 Compare Efficiency Factor with past races"):
        races = _pick("ef")
        if not races:
            return
        cur = {"d": _f(ef_delta), "m": _f(ef_mean), "1": _f(ef_1), "2": _f(ef_2)}
        rows = [_this_row(**{
            "Duration": duration_str(df),
            "Δ EF": _num(cur["d"], 2, signed=True), "EF mean": _num(cur["m"]),
            "EF 1st half": _num(cur["1"]), "EF 2nd half": _num(cur["2"]),
            "Verdict": verdict or "—", "Block (min)": block_min,
        })]
        for r in races:
            rows.append(_row(r, **{
                "Duration": _get(r, "race", "final_time") or "—",
                "Δ EF": _cell(_f(_get(r, "ef", "delta")), cur["d"], 2, signed=True),
                "EF mean": _cell(_f(_get(r, "ef", "race_mean")), cur["m"]),
                "EF 1st half": _cell(_f(_get(r, "ef", "first_half")), cur["1"]),
                "EF 2nd half": _cell(_f(_get(r, "ef", "second_half")), cur["2"]),
                "Verdict": _get(r, "ef", "verdict") or "—",
                "Block (min)": _get(r, "ef", "block_min"),
            }))
            pb = _get(r, "ef", "block_min")
            if pb is not None and pb != block_min:
                st.warning(f"⚠️ **{_label(r)}**: EF computed on {pb}-min blocks "
                           f"(this race: {block_min} min). Values are not strictly comparable.")
        table = pd.DataFrame(rows)
        table["Block (min)"] = table["Block (min)"].astype(str)
        _show(table)
        _pdf_toggle("ef", "Efficiency Factor", [("", table)])


# ======================================================================
# 7) TIME IN ZONE — colonna Total: % (heatmap) + durate h:mm
# ======================================================================
def _tiz_minutes_current(combined_df):
    if combined_df is None or "Total" not in combined_df.columns:
        return None
    return [hhmm_to_minutes(v) for v in combined_df["Total"].tolist()]


def _tiz_minutes_past(r):
    rows = [x for x in (r.get("time_in_zone") or []) if x.get("segment") == "Total"]
    if not rows:
        return None
    by_zone = {x.get("zone"): _f(x.get("minutes")) or 0 for x in rows}
    return [by_zone.get(z, 0) for z in HR_ZONE_LABELS]


def render_time_in_zone(combined_df, df):
    if not _past_races():
        return
    with st.expander("🔍 Compare time in zone with past races"):
        races = _pick("tiz")
        if not races:
            return
        cur_min = _tiz_minutes_current(combined_df)
        if not cur_min or not sum(cur_min):
            st.info("Time in zone of this race not available yet.")
            return
        cur_pct = [m / sum(cur_min) * 100 for m in cur_min]

        # colonne: (etichetta, durata gara, minuti per zona, % per zona, è la gara corrente)
        cols = [("This race", duration_str(df), cur_min, cur_pct, True)]
        for r in races:
            mins = _tiz_minutes_past(r)
            if not mins or not sum(mins):
                st.caption(f"{_label(r)}: no time-in-zone data in the curriculum.")
                continue
            cols.append((_short(r), _get(r, "race", "final_time") or "—", mins,
                         [m / sum(mins) * 100 for m in mins], False))

        x = [f"{c[0]}<br>race {c[1]}" for c in cols]
        z = np.array([c[3] for c in cols]).T
        text = []
        for c in cols:
            if c[4]:
                text.append([f"{p:.0f}%<br>{_hmm(m)}" for p, m in zip(c[3], c[2])])
            else:
                text.append([f"{p:.0f}%<br>{_hmm(m)}<br>Δ {round(cp - p) + 0:+d}"
                             for p, m, cp in zip(c[3], c[2], cur_pct)])
        fig = go.Figure(go.Heatmap(
            z=z, x=x, y=ZONES, text=np.array(text).T, texttemplate="%{text}",
            colorscale="YlOrRd", zmin=0, colorbar=dict(title="% time"),
            hovertemplate="%{y}: %{z:.0f}%<extra></extra>",
        ))
        fig.update_layout(title="Time in zone — % of total race time",
                          yaxis=dict(autorange="reversed"), height=400,
                          margin=dict(t=50, b=60, l=50, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Tabella: h:mm e % per zona, poi totale in zona e durata gara
        table = {"Zone": ZONES + ["Total in zones", "Race duration"]}
        for c in cols:
            vals = []
            for p, m, cp in zip(c[3], c[2], cur_pct):
                v = f"{_hmm(m)} · {p:.0f}%"
                vals.append(v if c[4] else f"{v} (Δ {round(cp - p) + 0:+d})")
            vals += [_hmm(sum(c[2])), c[1]]
            table[c[0]] = vals
        table = pd.DataFrame(table)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("Total race only. Compare the % (races of different length can't be "
                   "compared in minutes). Δ = this race − selected race, in percentage points.")
        _pdf_toggle("tiz", "Time in zone", [("", table)],
                    chart=("tiz_heatmap", {"cols": [f"{c[0]}\nrace {c[1]}" for c in cols],
                                           "z": z.tolist(), "zones": ZONES}))


# ======================================================================
# 5) CLIMBS
# ======================================================================
def _climb_info(row, pos=None):
    """Dati di una salita da una riga di tabella (corrente o curriculum)."""
    grade, vam, hr = _f(row.get("Avg Grade (%)")), _f(row.get("VAM (m/h)")), _f(row.get("Avg FC"))
    pcts = [row.get(f"% {z}") for z in ZONES]
    if pos is None:
        pos = _f(row.get("Position (%)"))
    return {
        "name": row.get("Climb Name", "?"), "grade": grade, "vam": vam, "hr": hr,
        "length": _f(row.get("Distance (km)")), "duration": row.get("Duration") or "—",
        "efs": _f(row.get("Avg EFS (km/h)")),
        "vam_bpm": vam / hr if vam is not None and hr else None,
        "pcts": pcts, "dominant": dominant_zone(pcts), "pos": pos,
    }


def _include_checkboxes(table, key_prefix):
    """Sotto il titolo: una checkbox per ogni salita delle gare passate.
    Le salite non spuntate spariscono dalla tabella (e dal PDF). La riga
    "This race" resta sempre. La chiave dipende dalle righe: se cambiano le
    gare scelte, le spunte ripartono tutte attive."""
    ident = "|".join(table["Race"].astype(str) + "/" + table["Climb"].astype(str))
    key = f"{key_prefix}_{hashlib.md5(ident.encode()).hexdigest()[:10]}"
    keep = []
    past = [i for i, r in table.iterrows() if r["Race"] != "▶ This race"]
    for start in range(0, len(past), 3):
        for col, i in zip(st.columns(3), past[start:start + 3]):
            label = f"{table.at[i, 'Race']} · {table.at[i, 'Climb']}"
            if col.checkbox(label, value=True, key=f"{key}_{i}"):
                keep.append(i)
    out = table[(table["Race"] == "▶ This race") | table.index.isin(keep)]
    out = out.reset_index(drop=True)
    _show(out)
    return out


def _pos_txt(p):
    return f"{p:.0f}%" if p is not None else "—"


def _similar_by(c, p):
    """'grade', 'length', 'grade + length' o None."""
    by = []
    if c["grade"] is not None and p["grade"] is not None and abs(p["grade"] - c["grade"]) <= GRADE_WINDOW:
        by.append("grade")
    if c["length"] and p["length"] is not None and abs(p["length"] - c["length"]) / c["length"] <= LENGTH_WINDOW:
        by.append("length")
    return " + ".join(by) or None


def render_climbs(climb_df, climb_data, df):
    if not _past_races() or climb_df is None or climb_df.empty:
        return
    with st.expander("🔍 Compare climbs with past races"):
        races = _pick("climbs")
        if not races:
            return
        pos = climb_positions(climb_data, df)
        all_current = [_climb_info(row, pos.get(row.get("Climb Name"), (None, None))[1])
                       for row in climb_df.to_dict(orient="records")]
        current = _checkbox_row(
            "Climbs of this race to compare:",
            [(_slug(c["name"]) + f"_{i}", c["name"], c) for i, c in enumerate(all_current)],
            "cmp_climb_sel")
        if not current:
            st.info("Select at least one climb.")
            return

        pdf_tables = []

        # --- A) Stessa salita, altre edizioni (solo nomi scritti a mano) ---
        st.markdown("**Same climb in other races** (matched by name — only names typed by hand)")
        same_rows = []
        for c in current:
            if CLIMB_AUTO_NAME_RE.match(str(c["name"])):
                continue
            matches = [(r, _climb_info(row)) for r in races
                       for row in (r.get("climbs") or [])
                       if _slug(row.get("Climb Name")) == _slug(c["name"])]
            if not matches:
                continue
            same_rows.append({"Race": "▶ This race", "Climb": c["name"],
                              "Length (km)": _num(c["length"], 1), "Duration": c["duration"],
                              "VAM (m/h)": _num(c["vam"], 0), "Avg HR (bpm)": _num(c["hr"], 0),
                              "EFS (km/h)": _num(c["efs"]),
                              **{f"% {z}": c["pcts"][i] or "—" for i, z in enumerate(ZONES)},
                              "Position": _pos_txt(c["pos"])})
            for r, p in matches:
                same_rows.append({"Race": _short(r), "Climb": p["name"],
                                  "Length (km)": _num(p["length"], 1), "Duration": p["duration"],
                                  "VAM (m/h)": _cell(p["vam"], c["vam"], 0),
                                  "Avg HR (bpm)": _cell(p["hr"], c["hr"], 0),
                                  "EFS (km/h)": _cell(p["efs"], c["efs"]),
                                  **{f"% {z}": p["pcts"][i] or "—" for i, z in enumerate(ZONES)},
                                  "Position": _pos_txt(p["pos"])})
        if same_rows:
            t = _include_checkboxes(pd.DataFrame(same_rows), "cmp_same")
            pdf_tables.append(("Same climb in other races", t))
        else:
            st.caption("No climb with the same name in the selected races.")

        # --- B) Salite simili: pendenza ±2 punti OPPURE lunghezza ±40% ---
        st.markdown(f"**Similar climbs** (grade ±{GRADE_WINDOW} percentage points "
                    f"or length ±{LENGTH_WINDOW * 100:.0f}%)")
        any_similar = False
        for c in current:
            rows = [{"Race": "▶ This race", "Climb": c["name"], "Similar by": "—",
                     "Length (km)": _num(c["length"], 1), "Duration": c["duration"],
                     "Grade (%)": _num(c["grade"], 0), "VAM (m/h)": _num(c["vam"], 0),
                     "Avg HR (bpm)": _num(c["hr"], 0), "VAM/bpm": _num(c["vam_bpm"]),
                     "EFS (km/h)": _num(c["efs"]), "Main zone": c["dominant"],
                     "Position": _pos_txt(c["pos"])}]
            for r in races:
                for row in r.get("climbs") or []:
                    p = _climb_info(row)
                    by = _similar_by(c, p)
                    if not by:
                        continue
                    rows.append({"Race": _short(r), "Climb": p["name"], "Similar by": by,
                                 "Length (km)": _cell(p["length"], c["length"], 1),
                                 "Duration": p["duration"],
                                 "Grade (%)": _num(p["grade"], 0),
                                 "VAM (m/h)": _cell(p["vam"], c["vam"], 0),
                                 "Avg HR (bpm)": _cell(p["hr"], c["hr"], 0),
                                 "VAM/bpm": _cell(p["vam_bpm"], c["vam_bpm"]),
                                 "EFS (km/h)": _cell(p["efs"], c["efs"]),
                                 "Main zone": p["dominant"], "Position": _pos_txt(p["pos"])})
            if len(rows) > 1:
                any_similar = True
                sub = (f"{c['name']} ({_num(c['length'], 1)} km, {_num(c['grade'], 0)}%) — "
                       "similar climbs")
                st.markdown(f"⛰️ {sub}")
                t = _include_checkboxes(pd.DataFrame(rows), "cmp_sim")
                pdf_tables.append((sub, t))
        if not any_similar:
            st.caption("No similar climb in the selected races.")
        st.caption("Position = start of the climb in % of race time (— for races saved "
                   "before this field existed). Main zone = HR zone with the most time.")
        if pdf_tables:
            _pdf_toggle("climbs", "Climbs", pdf_tables)


# ======================================================================
# 6) LAPS — Lap 1 contro Lap 1, Lap 2 contro Lap 2, ...
# ======================================================================
LAP_DELTA_COLS = {"Distance (km)": 1, "Elevation (m)": 0, "Avg FC": 0,
                  "Speed (km/h)": 1, "Avg EFS (km/h)": 2}


def render_laps(lap_df):
    if not _past_races() or lap_df is None or lap_df.empty:
        return
    with st.expander("🔍 Compare laps with past races"):
        races = _pick("laps")
        if not races:
            return
        visible = st.session_state.get("lap_visible_cols") or list(lap_df.columns)
        name_col = next((c for c in visible if c.endswith("Name")), None)
        cur_laps = lap_df.to_dict(orient="records")
        past = [(r, r.get("laps") or []) for r in races]
        n_max = max([len(cur_laps)] + [len(l) for _, l in past])
        chosen = _checkbox_row("Laps to compare:",
                               [(str(n), f"Lap {n}", n) for n in range(1, n_max + 1)],
                               "cmp_lap_sel")
        if not chosen:
            st.info("Select at least one lap.")
            return

        cols = [c for c in visible if c != name_col]
        rows = []
        for n in chosen:
            cur = cur_laps[n - 1] if n <= len(cur_laps) else None
            if cur is not None:
                rows.append({"Lap #": n, "Race": "▶ This race",
                             "Lap": cur.get(name_col, f"Lap {n}"),
                             **{c: str(cur.get(c, "—")) for c in cols}})
            for r, laps in past:
                if n > len(laps):
                    continue
                lap = laps[n - 1]
                row = {"Lap #": n, "Race": _short(r), "Lap": lap.get(name_col, f"Lap {n}")}
                for c in cols:
                    v = lap.get(c)
                    if c in LAP_DELTA_COLS and cur is not None:
                        row[c] = _cell(_f(v), _f(cur.get(c)), LAP_DELTA_COLS[c])
                    else:
                        row[c] = "—" if v is None else str(v)
                rows.append(row)
        table = pd.DataFrame(rows)
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("Each lap against the lap with the same number in the other races. "
                   "Same columns as the lap table above. " + DELTA_CAPTION)

        # Nel PDF: dati del lap e zone in due tabelle, una sola sarebbe illeggibile
        zone_cols = [c for c in table.columns if c in ZONES or c.startswith("% Z")]
        info_cols = [c for c in table.columns if c not in zone_cols]
        pdf_tables = [("", table[info_cols])]
        if zone_cols:
            pdf_tables.append(("Time in zone", table[["Lap #", "Race"] + zone_cols]))
        _pdf_toggle("laps", "Laps", pdf_tables)