"""11) Confronto con le gare del curriculum.

In cima alla pagina (render_top): elenco delle gare del curriculum e scelta
delle gare da confrontare. Quella scelta diventa il default di ogni sezione.

Ogni sezione dell'analisi chiama la sua funzione render_* qui sotto, che
disegna un expander con:
  - le gare da confrontare (precompilate con la scelta in cima, modificabili);
  - i parametri da mostrare (tutti di default);
  - per ogni parametro una riga: valore di questa gara, "dot strip" con le
    gare scelte (grigie) e questa gara (blu), pillole di verdetto contro OGNI
    gara scelta (verde = meglio, rosso = peggio, grigio = senza direzione);
  - opzionale: trend nel tempo (gare scelte + questa, in ordine di data);
  - opzionale: tabella con i numeri esatti;
  - "Add this comparison to the PDF": il PDF riceve esattamente ciò che si
    vede (parametri scelti, trend e tabella solo se attivi).

Se nessun curriculum è caricato in cima alla pagina, non viene mostrato nulla.
Questo modulo non importa nessun file di sezione (niente import circolari).
"""
import hashlib
import html
import math

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from climbs import CLIMB_AUTO_NAME_RE
from curriculum import (
    _slug, apply_settings, climb_positions, load_curriculum, race_id, races_table,
)
from input_data import HR_ZONE_LABELS, hhmm_to_minutes


# ======================================================================
# HELPERS
# ======================================================================
TOLERANCE = 0.20          # ±20% su distanza e D+ → avviso di confrontabilità
GRADE_WINDOW = 2          # salite simili: pendenza ±2 punti percentuali
LENGTH_WINDOW = 0.40      # salite simili: lunghezza ±40%
ZONES = ["Z1", "Z2", "Z3", "Z4", "Z5"]
SECTIONS = ["total", "detspd", "ef", "tiz", "climbs", "laps"]

# Direzione di ogni parametro → colore delle pillole
#   up      più alto = meglio          down  più basso = meglio
#   zero    più vicino a 0 = meglio    none  nessuna direzione (grigio)
#   context descrive il percorso, non la prestazione (grigio, corsivo)
DIR_HINT = {"up": "higher is better", "down": "lower is better",
            "zero": "closer to 0 is better", "none": "", "context": "course, not performance"}
# Banda di tolleranza: dentro la banda la differenza è "uguale" (grigio).
# ("pct", x) = ±x% del valore della gara selezionata · ("abs", x) = ±x in unità del parametro
TOL = {
    "Avg EFS (km/h)": ("pct", 4), "HR 2nd vs 1st half (%)": ("abs", 4),
    "DET index": ("abs", 2.0), "HR drift (bpm/h)": ("abs", 2.0),
    "SPD index": ("abs", 0.4), "Speed decay (km/h per h)": ("abs", 0.10),
    "Δ EF": ("abs", 0.40), "EF mean": ("pct", 4), "EF 1st half": ("pct", 4), "EF 2nd half": ("pct", 4),
    "VAM (m/h)": ("pct", 10), "VAM/bpm": ("pct", 10), "EFS (km/h)": ("pct", 6),
    "Speed (km/h)": ("pct", 6), "Avg EFS (km/h) lap": ("pct", 6), "Pace (min/km)": ("pct", 6),
    "zone": ("abs", 6),
}


def _tol_txt(tol):
    if not tol:
        return ""
    kind, x = tol
    return f"±{x:g}% = same" if kind == "pct" else f"±{x:g} = same"


GOOD, BAD, NEUTRAL = "#0ca30c", "#d03b3b", "#888780"
GOOD_LIGHT, BAD_LIGHT = "#7fcf7f", "#ee9a9a"   # oltre la banda ma entro 2× banda
MIXED = "#e8a317"                 # meglio di alcune gare, peggio di altre
THIS_COLOR = "#2a78d6"            # parametri senza direzione / valori uguali
# Tipi di verdetto: good/bad = netto (oltre 2× banda), good_light/bad_light = lieve
DOT_COLOR = {"good": GOOD, "good_light": GOOD_LIGHT, "bad": BAD, "bad_light": BAD_LIGHT,
             "mixed": MIXED, "neutral": THIS_COLOR}
DATE_FALLBACK = "—"


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


def _hmm(minutes):
    if minutes is None:
        return "—"
    m = int(round(minutes))
    return f"{m // 60}:{m % 60:02d}"


def _pace_to_sec(p):
    """'06:49' → 409 secondi/km."""
    try:
        m, s = str(p).split(":")
        return int(m) * 60 + int(s)
    except (ValueError, AttributeError):
        return None


def _sec_to_pace(s):
    if s is None:
        return "—"
    s = int(round(s))
    return f"{s // 60:02d}:{s % 60:02d}"


def _hms_to_sec(t):
    """'0:35:00' o '1:30' → secondi."""
    try:
        parts = [int(x) for x in str(t).split(":")]
    except ValueError:
        return None
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 3600 + parts[1] * 60
    return None


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


SOURCES = ["This athlete", "Another athlete", "Both"]


def _external_races():
    """Gare del curriculum di un altro atleta, marcate come esterne."""
    ext = st.session_state.get("cmp_ext_cv")
    if not ext:
        return []
    out = []
    for r in ext.get("races") or []:
        r = dict(r)
        r["_athlete"] = ext.get("athlete_name", "?")
        r["race_id"] = f"ext:{ext.get('athlete_id', '')}:{r.get('race_id')}"
        out.append(r)
    return out


def _past_races():
    """Gare confrontabili secondo la sorgente scelta in cima, esclusa la gara
    che si sta analizzando."""
    ss = st.session_state
    source = ss.get("cmp_source", SOURCES[0])
    own = (ss.get("curriculum_in") or {}).get("races") or []
    races = []
    if source in ("This athlete", "Both"):
        this_id = None
        if ss.get("race_name") and ss.get("race_date") is not None:
            this_id = race_id(ss["race_name"], ss["race_date"].isoformat())
        races += [r for r in own if r.get("race_id") != this_id]
    if source in ("Another athlete", "Both"):
        races += _external_races()
    return races


def _label(r):
    who = f"{r['_athlete']} · " if r.get("_athlete") else ""
    return f"{who}{_get(r, 'race', 'date') or '?'} · {_get(r, 'race', 'name') or '?'}"


def _short(r):
    """'Trail del Monte (2025)', con il nome dell'atleta se è di un altro atleta."""
    d = _get(r, "race", "date") or ""
    who = f"{r['_athlete']} · " if r.get("_athlete") else ""
    base = f"{_get(r, 'race', 'name') or '?'} ({d[:4]})" if d else (_get(r, "race", "name") or "?")
    return who + base


def _ext_prefixes(races):
    """Prefissi delle etichette delle gare di altri atleti (per nascondere la FC assoluta)."""
    return tuple(_short(r) for r in races if r.get("_athlete"))


def _date_this():
    d = st.session_state.get("race_date")
    return d.isoformat() if d is not None else DATE_FALLBACK


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
    """Multiselect delle gare, precompilato con la scelta fatta in cima."""
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
    if any(r.get("_athlete") for r in races):
        st.info("ℹ️ Other athletes' races: absolute heart-rate data (Avg HR, HR drift, "
                "VAM/bpm) is hidden for them. Only data scaled to the threshold, time in zone "
                "and speed data are compared.")
    st.session_state["_cmp_ext_now"] = _ext_prefixes(races)
    return races


def _params(section, labels):
    """Multiselect dei parametri da mostrare (tutti di default)."""
    key = f"cmp_{section}_params"
    ss = st.session_state
    if key in ss:
        ss[key] = [p for p in ss[key] if p in labels]
    else:
        ss[key] = list(labels)
    return st.multiselect("Parameters", labels, key=key,
                          placeholder="Choose the parameters to compare")


def _options(section):
    """Toggle trend e tabella numeri. Ritorna (show_trend, show_numbers)."""
    c1, c2 = st.columns(2)
    trend = c1.toggle("📈 Show trend over time", key=f"cmp_{section}_trend")
    numbers = c2.toggle("🔢 Show the numbers", key=f"cmp_{section}_numbers")
    return trend, numbers


def _pdf_toggle(section, title, blocks, heatmap=None, races=None):
    """Checkbox "Add to PDF": il PDF riceve esattamente quello che si vede.
    races: gare usate nel confronto, evidenziate nel PDF sotto il titolo."""
    if st.checkbox("📄 Add this comparison to the PDF", key=f"cmp_{section}_pdf"):
        st.session_state.setdefault("cmp_pdf", {})[section] = {
            "title": title, "blocks": blocks, "heatmap": heatmap,
            "races": [_label(r) for r in (races or [])],
        }


def _checkbox_grid(items, key, per_row=3):
    """Checkbox (spuntate di default) per una lista di (etichetta, oggetto)."""
    ident = "|".join(lbl for lbl, _ in items)
    base = f"{key}_{hashlib.md5(ident.encode()).hexdigest()[:10]}"
    chosen = []
    for start in range(0, len(items), per_row):
        for col, (i, (lbl, obj)) in zip(st.columns(per_row),
                                          list(enumerate(items))[start:start + per_row]):
            if col.checkbox(lbl, value=True, key=f"{base}_{i}"):
                chosen.append(obj)
    return chosen


# ======================================================================
# COMPONENTE: righe parametro (dot strip + pillole + trend)
# ======================================================================
def metric(label, cur, past, direction="none", nd=2, unit="", fmt=None, hr_abs=False, tol=None):
    """Una riga di confronto.
    past = [(nome breve, data iso, valore)] — una voce per gara/salita/lap.
    hr_abs=True: dato di FC assoluta → tolto per le gare di altri atleti; se non
    resta nessuna gara, la riga viene nascosta (drop)."""
    fmt = fmt or (lambda v: _num(v, nd))
    drop = False
    if hr_abs:
        ext = st.session_state.get("_cmp_ext_now") or ()
        kept = [p for p in past if not (ext and str(p[0]).startswith(ext))]
        drop = bool(past) and not kept
        past = kept
    # tutto alla precisione mostrata: strip, trend e verdetti vedono gli stessi numeri
    r = lambda v: None if v is None else round(float(v), nd) + 0.0
    cur = r(cur)
    past = [(n, d, r(v)) for n, d, v in past]
    if tol is None and direction not in ("none", "context"):
        tol = TOL.get(label)
    return {"label": label, "unit": unit, "direction": direction, "nd": nd, "tol": tol,
            "cur": cur, "cur_fmt": fmt(cur), "fmt": fmt, "past": past, "drop": drop}


def _within_tol(tol, a, b, factor=1):
    """True se a e b differiscono meno di factor × banda di tolleranza."""
    if not tol:
        return False
    kind, x = tol
    if kind == "pct":
        return b != 0 and abs(a - b) / abs(b) * 100 <= x * factor
    return abs(a - b) <= x * factor


def _verdict(direction, cur, past, nd=2, tol=None):
    """'good' | 'bad' | 'neutral' per questa gara rispetto a una gara passata.
    Confronto alla precisione mostrata (9.871 e 9.87 sono uguali) e dentro la
    banda di tolleranza del parametro la differenza conta come "uguale"."""
    if cur is None or past is None or direction in ("none", "context"):
        return "neutral"
    if direction == "zero":
        a, b = round(abs(cur), nd), round(abs(past), nd)
    else:
        a, b = round(cur, nd), round(past, nd)
    if a == b or _within_tol(tol, a, b):
        return "neutral"
    better = a > b if direction == "up" else a < b
    kind = "good" if better else "bad"
    # lieve: oltre la banda ma entro il doppio; netto: oltre il doppio
    return kind + "_light" if _within_tol(tol, a, b, factor=2) else kind


def _delta_txt(m, d):
    """Δ formattato come il parametro: passo in s/km, durate in h:mm:ss."""
    if m["fmt"] is _sec_to_pace:
        return f"{d:+.0f} s/km"
    if m["fmt"] is _hms:
        return ("+" if d >= 0 else "-") + _hms(abs(d))
    return f"{round(d, m['nd']) + 0.0:+.{m['nd']}f}{m['unit']}"


def _pills(m):
    """[(testo, tipo)] — una pillola per ogni gara passata."""
    out = []
    for name, _, v in m["past"]:
        if v is None or m["cur"] is None:
            out.append((f"— vs {name}", "neutral"))
            continue
        d = round(m["cur"], m["nd"]) - round(v, m["nd"])
        if d == 0:
            out.append((f"= same as {name}", "neutral"))
            continue
        kind = _verdict(m["direction"], m["cur"], v, m["nd"], m.get("tol"))
        a, b = (abs(m["cur"]), abs(v)) if m["direction"] == "zero" else (m["cur"], v)
        within = m["direction"] not in ("none", "context") and _within_tol(m.get("tol"), a, b)
        arrow = "≈" if within else ("▲" if d > 0 else "▼")
        out.append((f"{arrow} {_delta_txt(m, d)} vs {name}", kind))
    return out


def cur_kind(pills):
    """Colore del pallino di questa gara dai verdetti contro ogni gara:
    tutti meglio → good (netto se tutti netti, altrimenti lieve), tutti peggio →
    bad (idem), meglio e peggio insieme → mixed, nessuna differenza → neutral."""
    kinds = {k for _, k in pills if k != "neutral"}
    if not kinds:
        return "neutral"
    sides = {k.split("_")[0] for k in kinds}
    if len(sides) > 1:
        return "mixed"
    side = sides.pop()
    return side if kinds == {side} else side + "_light"


def band_unit(m):
    """Ampiezza della banda di tolleranza in unità del parametro (None se non c'è)."""
    tol, cur = m.get("tol"), m.get("cur")
    if not tol or cur is None:
        return None
    kind, x = tol
    unit = abs(cur) * x / 100 if kind == "pct" else x
    return unit or None


def block_span(metrics):
    """Semi-ampiezza comune delle strip, in bande: almeno ±3 bande, o quanto serve
    per far stare la differenza più grande del blocco. Così la stessa distanza
    vuol dire la stessa cosa su tutte le righe."""
    span = 3.0
    for m in metrics:
        unit = band_unit(m)
        if unit:
            for _, _, v in m["past"]:
                if v is not None:
                    span = max(span, abs(v - m["cur"]) / unit * 1.1)
    return span


def strip_layout(m, span):
    """Posizioni 0..1 sulla strip. Con banda: questa gara al centro, distanze in
    bande (scala comune del blocco), zona ±1 banda e tacche a ±2 bande.
    Senza banda: scala propria min-max, come prima."""
    unit = band_unit(m)
    if unit and span:
        x = lambda v: 0.5 + (v - m["cur"]) / unit / (2 * span)
        return {"past": [None if v is None else x(v) for _, _, v in m["past"]],
                "cur": 0.5, "band": (0.5 - 1 / (2 * span), 0.5 + 1 / (2 * span)),
                "ticks": [0.5 - 2 / (2 * span), 0.5 + 2 / (2 * span)]}
    vals = [v for _, _, v in m["past"] if v is not None] + ([m["cur"]] if m["cur"] is not None else [])
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.08 or (abs(hi) * 0.05 or 1)
    lo, hi = lo - pad, hi + pad
    x = lambda v: (v - lo) / (hi - lo)
    return {"past": [None if v is None else x(v) for _, _, v in m["past"]],
            "cur": None if m["cur"] is None else x(m["cur"]), "band": None, "ticks": []}


def _strip_svg(m, span=None):
    """Dot strip: grigi = gare scelte, pallino grande = questa gara."""
    lay = strip_layout(m, span)
    if lay is None:
        return "<div style='font-size:12px;opacity:.6'>no data</div>"
    X = lambda f: 12 + f * 276
    parts = ['<svg viewBox="0 0 300 46" width="100%" height="46">']
    if lay["band"]:
        b0, b1 = X(lay["band"][0]), X(lay["band"][1])
        parts.append(f'<rect x="{b0:.1f}" y="14" width="{b1 - b0:.1f}" height="16" rx="3" '
                     f'fill="{THIS_COLOR}" fill-opacity="0.12"><title>±1 tolerance band = same'
                     f'</title></rect>')
        for t in lay["ticks"]:
            parts.append(f'<line x1="{X(t):.1f}" y1="16" x2="{X(t):.1f}" y2="28" stroke="{NEUTRAL}" '
                         f'stroke-opacity="0.6" stroke-dasharray="2 2"/>')
    parts.append(f'<line x1="6" y1="22" x2="294" y2="22" stroke="{NEUTRAL}" stroke-opacity="0.5"/>')
    labels = len(m["past"]) <= 4
    for i, ((name, _, v), f) in enumerate(zip(m["past"], lay["past"])):
        if v is None:
            continue
        cx = X(min(max(f, 0), 1))
        t = html.escape(f"{name}: {m['fmt'](v)}")
        parts.append(f'<circle cx="{cx:.1f}" cy="22" r="5" fill="{NEUTRAL}"><title>{t}</title></circle>')
        if labels:
            y = 41 if i % 2 == 0 else 10
            parts.append(f'<text x="{cx:.1f}" y="{y}" text-anchor="middle" font-size="10" '
                         f'fill="currentColor" fill-opacity="0.65">{html.escape(m["fmt"](v))}</text>')
    if lay["cur"] is not None:
        col = DOT_COLOR[cur_kind(_pills(m))]
        parts.append(f'<circle cx="{X(lay["cur"]):.1f}" cy="22" r="8" fill="{col}" stroke="white" '
                     f'stroke-width="2"><title>This race: {html.escape(m["cur_fmt"])}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def _trend_svg(m, this_date):
    """Trend: gare scelte + questa, in ordine di data."""
    pts = [(d, v, False) for _, d, v in m["past"] if v is not None and d]
    if m["cur"] is not None and this_date and this_date != DATE_FALLBACK:
        pts.append((this_date, m["cur"], True))
    pts.sort(key=lambda p: p[0])
    if len(pts) < 2:
        return ""
    ds = pd.to_datetime([p[0] for p in pts])
    t0, t1 = ds.min(), ds.max()
    span = (t1 - t0).total_seconds() or 1
    vals = [p[1] for p in pts]
    lo, hi = min(vals), max(vals)
    pad = (hi - lo) * 0.15 or (abs(hi) * 0.05 or 1)
    lo, hi = lo - pad, hi + pad
    xs = [12 + (d - t0).total_seconds() / span * 276 for d in ds]
    ys = [40 - (v - lo) / (hi - lo) * 32 for v in vals]
    poly = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xs, ys))
    parts = ['<svg viewBox="0 0 300 56" width="100%" height="56">',
             f'<polyline points="{poly}" fill="none" stroke="{NEUTRAL}" stroke-width="1.5"/>']
    this_col = DOT_COLOR[cur_kind(_pills(m))]
    for (d, v, is_cur), a, b in zip(pts, xs, ys):
        col, r = (this_col, 6) if is_cur else (NEUTRAL, 4)
        parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="{r}" fill="{col}">'
                     f'<title>{html.escape(d)}: {html.escape(m["fmt"](v))}</title></circle>')
    parts.append(f'<text x="12" y="54" font-size="10" fill="currentColor" fill-opacity="0.6">{t0:%m/%Y}</text>')
    parts.append(f'<text x="288" y="54" font-size="10" text-anchor="end" fill="currentColor" '
                 f'fill-opacity="0.6">{t1:%m/%Y}</text></svg>')
    return "".join(parts)


_PILL_STYLE = {
    "good": f"background:rgba(12,163,12,0.30);color:{GOOD};font-weight:600",
    "good_light": f"background:rgba(12,163,12,0.10);color:{GOOD}",
    "bad": f"background:rgba(208,59,59,0.30);color:{BAD};font-weight:600",
    "bad_light": f"background:rgba(208,59,59,0.10);color:{BAD}",
    "neutral": f"background:rgba(42,120,214,0.15);color:{THIS_COLOR}",   # blu come il pallino
}


def render_metrics(metrics, trend=False):
    """Disegna le righe parametro in un unico blocco HTML."""
    this_date = _date_this()
    span = block_span(metrics)
    rows = []
    for m in metrics:
        hint = DIR_HINT.get(m["direction"], "")
        if m.get("tol"):
            hint += f" · {_tol_txt(m['tol'])}"
        italic = "font-style:italic;" if m["direction"] == "context" else ""
        pills = "".join(
            f"<span style='display:inline-block;font-size:12px;padding:2px 8px;border-radius:6px;"
            f"margin:2px 4px 2px 0;{_PILL_STYLE[k]}'>{html.escape(t)}</span>"
            for t, k in _pills(m))
        graph = _strip_svg(m, span) + (_trend_svg(m, this_date) if trend else "")
        rows.append(
            "<div style='display:grid;grid-template-columns:170px 80px minmax(0,1fr) 260px;"
            "gap:14px;align-items:center;padding:8px 0;border-bottom:1px solid rgba(128,128,128,0.2)'>"
            f"<div style='font-size:13px;{italic}'>{html.escape(m['label'])}"
            f"<br><span style='font-size:11px;opacity:.6'>{hint}</span></div>"
            f"<div style='font-size:20px;font-weight:500;text-align:right;{italic}'>"
            f"{html.escape(m['cur_fmt'])}</div>"
            f"<div>{graph}</div><div>{pills}</div></div>")
    legend = (f"<div style='font-size:12px;opacity:.7;margin:4px 0 2px'>"
              f"Big dot = this race: <span style='color:{GOOD}'>●</span> clearly / "
              f"<span style='color:{GOOD_LIGHT}'>●</span> slightly better than all · "
              f"<span style='color:{BAD}'>●</span> clearly / "
              f"<span style='color:{BAD_LIGHT}'>●</span> slightly worse than all · "
              f"<span style='color:{MIXED}'>●</span> mixed · "
              f"<span style='color:{THIS_COLOR}'>●</span> no direction / same &nbsp;|&nbsp; "
              f"<span style='color:{NEUTRAL}'>●</span> selected races &nbsp;|&nbsp; "
              f"Δ = this race − selected · ≈ = within the band (blue) · strong colour = "
              f"beyond 2× the band · shaded zone = ±1 band, dashed ticks = ±2 bands "
              f"(same scale on every row)</div>")
    st.markdown(legend + "".join(rows), unsafe_allow_html=True)


def numbers_table(metrics, names):
    """Valori esatti: una riga per gara, valore e Δ in colonne separate."""
    rows = [{"Race": "▶ This race", **{m["label"]: m["cur_fmt"] for m in metrics}}]
    for i, name in enumerate(names):
        row = {"Race": name}
        for m in metrics:
            v = next((pv for pn, _, pv in m["past"] if pn == name), None)
            row[m["label"]] = m["fmt"](v) if v is not None else "—"
            if m["direction"] != "context":
                d = (round(m["cur"], m["nd"]) - round(v, m["nd"])
                     if (v is not None and m["cur"] is not None) else None)
                row[f"Δ {m['label']}"] = "—" if d is None else _delta_txt(m, d)
        rows.append(row)
    return pd.DataFrame(rows).fillna("")


def _block(metrics, trend, numbers, names, subtitle="", chips="", context=None):
    """Disegna un blocco completo e ritorna la sua copia per il PDF.
    context: tabella già mostrata sopra (solo per il PDF)."""
    metrics = [m for m in metrics if not m.get("drop")]
    if chips:
        st.caption(chips)
    render_metrics(metrics, trend)
    table = None
    if numbers:
        table = numbers_table(metrics, names)
        st.dataframe(table, hide_index=True, use_container_width=True)
    span = block_span(metrics)
    pdf_rows = [{"label": m["label"], "direction": m["direction"], "cur": m["cur"],
                 "cur_fmt": m["cur_fmt"], "past": [(n, d, v, m["fmt"](v) if v is not None else "—")
                                                    for n, d, v in m["past"]],
                 "pills": _pills(m), "layout": strip_layout(m, span)} for m in metrics]
    return {"subtitle": subtitle, "chips": chips, "rows": pdf_rows, "context": context,
            "trend": trend, "this_date": _date_this(), "table": table}


# ======================================================================
# IN CIMA ALLA PAGINA: gare del curriculum + scelta di default
# ======================================================================
def render_top():
    ss = st.session_state
    cur = ss.get("curriculum_in")
    if cur and cur.get("races"):
        st.markdown("**Races in the curriculum**")
        st.dataframe(races_table(cur), hide_index=True, use_container_width=True)
        _reopen(cur)

    # --- Con chi confrontare: questo atleta, un altro, entrambi ---
    st.markdown("**🔍 Compare this race with**")
    source = st.radio("Compare with races of", SOURCES, key="cmp_source", horizontal=True,
                      label_visibility="collapsed")
    if source in ("Another athlete", "Both"):
        up = st.file_uploader("Curriculum of another athlete (.json)", type=["json"],
                              key="cmp_ext_upload")
        if up is None:
            ss.pop("cmp_ext_cv", None)
        else:
            try:
                ext = load_curriculum(up)
                ss["cmp_ext_cv"] = ext
                st.success(f"✅ {ext.get('athlete_name')}: {len(ext.get('races', []))} race(s) "
                           "available for comparison.")
            except ValueError as e:
                ss.pop("cmp_ext_cv", None)
                st.error(f"❌ {e}")

    races = _past_races()
    if races:
        by_id = {r["race_id"]: r for r in races}
        if "cmp_default_races" in ss:
            ss["cmp_default_races"] = [i for i in ss["cmp_default_races"] if i in by_id]
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
    elif source == "This athlete" and not cur:
        st.caption("Upload this athlete's curriculum above, or choose another athlete.")
    st.markdown("---")


def _reopen(cur):
    """Riapre un'analisi salvata: si sceglie la gara, poi si carica il suo FIT."""
    ss = st.session_state
    with st.expander("↩️ Reopen a past analysis"):
        by_id = {r["race_id"]: r for r in cur["races"]}
        rid = st.selectbox("Race to reopen", list(by_id),
                           format_func=lambda i: _label(by_id[i]), key="reopen_race")
        if st.button("Reopen this race", key="reopen_btn"):
            full = apply_settings(by_id[rid])
            ss["_reopened"] = {"race_id": rid, "km": _get(by_id[rid], "race", "distance_km"),
                               "date": _get(by_id[rid], "race", "date"),
                               "label": _label(by_id[rid]),
                               "old": not by_id[rid].get("settings"), "partial": not full}
        info = ss.get("_reopened")
        if info:
            msg = (f"↩️ **{info['label']}** reopened: race info, HR zones and coach comment are "
                   "filled in. Now upload the **.fit / .fit.gz of this race** in the Race File "
                   "section below.")
            if info.get("old"):
                msg += (" This race was saved with an older version: segments, climbs and laps "
                        "were rebuilt from its saved results (laps one after the other from the "
                        "start), so give them a quick check.")
                if info.get("partial"):
                    msg += " Climbs saved before their start time was stored must be set again."
            st.success(msg)


# ======================================================================
# Sezioni a livello gara (Total, DET & SPD, EF): stessa struttura
# ======================================================================
def _race_section(section, title, expander_label, build_metrics, chips_fn=None, extra=None):
    if not _past_races():
        return
    with st.expander(expander_label):
        races = _pick(section)
        if not races:
            return
        all_metrics = build_metrics(races)
        chosen = _params(section, [m["label"] for m in all_metrics])
        metrics = [m for m in all_metrics if m["label"] in chosen]
        if not metrics:
            st.info("Choose at least one parameter.")
            return
        trend, numbers = _options(section)
        if extra:
            extra(races)
        chips = chips_fn(races) if chips_fn else ""
        block = _block(metrics, trend, numbers, [_short(r) for r in races], chips=chips)
        _pdf_toggle(section, title, [block], races=races)


def _past(races, *path, conv=_f, neg=False):
    out = []
    for r in races:
        v = conv(_get(r, *path))
        if neg and v is not None:
            v = -v
        out.append((_short(r), _get(r, "race", "date"), v))
    return out


def _course_chips(races, df=None):
    """'This race: 157 km · 3760 D+ · 15:52:08 | Vipava (2025): 112 km · ...'"""
    ss = st.session_state
    def one(name, km, dp, t):
        dpk = f" · {dp / km:.0f} m/km" if km and dp is not None else ""
        return f"{name}: {_num(km, 1)} km · {_num(dp, 0)} D+ · {t}{dpk}"
    parts = [one("This race", _f(ss.get("kilometers")), _f(ss.get("total_elevation_gain")),
                 duration_str(df) if df is not None else "—")]
    for r in races:
        parts.append(one(_short(r), _f(_get(r, "race", "distance_km")),
                         _f(_get(r, "race", "elevation_gain_m")), _get(r, "race", "final_time") or "—"))
    return "  |  ".join(parts)


# ----------------------------------------------------------------------
# 1) TOTAL DATA
# ----------------------------------------------------------------------
def render_total(df, overall_avg, percent_diff, avg_efs_fn):
    """avg_efs_fn() calcola l'EFS medio della gara corrente solo se serve."""
    ss = st.session_state

    def build(races):
        km, dp = _f(ss.get("kilometers")), _f(ss.get("total_elevation_gain"))
        past_dpk = []
        for r in races:
            pkm, pdp = _f(_get(r, "race", "distance_km")), _f(_get(r, "race", "elevation_gain_m"))
            past_dpk.append((_short(r), _get(r, "race", "date"),
                             pdp / pkm if pkm and pdp is not None else None))
        return [
            metric("Avg EFS (km/h)", _f(avg_efs_fn()), _past(races, "efs", "avg_efs_kmh"), "up", 2),
            metric("Avg HR (bpm)", _f(overall_avg), _past(races, "heart_rate", "overall_avg"), "none", 0,
                   hr_abs=True),
            metric("HR 2nd vs 1st half (%)", _f(percent_diff),
                   _past(races, "heart_rate", "percent_diff"), "zero", 1,
                   fmt=lambda v: _num(v, 1, signed=True)),
            metric("D+/km (m/km)", dp / km if km and dp is not None else None, past_dpk, "context", 0),
            metric("Distance (km)", km, _past(races, "race", "distance_km"), "context", 1),
            metric("D+ (m)", dp, _past(races, "race", "elevation_gain_m"), "context", 0),
        ]

    _race_section("total", "Total data", "🔍 Compare total data with past races", build,
                  chips_fn=lambda races: _course_chips(races, df))


# ----------------------------------------------------------------------
# 2) DET + SPD
# ----------------------------------------------------------------------
def render_det_spd(df, det_index, hr_drift, det_verdict, spd_index, spd_kmh_per_h, spd_verdict):
    """Speed decay mostrato come nel badge: negativo = rallenta."""
    if _past_races():
        # spazio fra i badge DET/SPD e l'expander (altrimenti si toccano)
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    cur_dec = -_f(spd_kmh_per_h) if _f(spd_kmh_per_h) is not None else None

    def build(races):
        return [
            metric("DET index", _f(det_index), _past(races, "det", "index"), "down", 1),
            metric("HR drift (bpm/h)", _f(hr_drift), _past(races, "det", "hr_drift_bpm_per_h"),
                   "up", 2, fmt=lambda v: _num(v, 2, signed=True), hr_abs=True),
            metric("SPD index", _f(spd_index), _past(races, "spd", "index"), "down", 2),
            metric("Speed decay (km/h per h)", cur_dec,
                   _past(races, "spd", "speed_decay_kmh_per_h", neg=True), "zero", 3,
                   fmt=lambda v: _num(v, 3, signed=True)),
        ]

    def chips(races):
        parts = [f"This race: {duration_str(df)} · DET {det_verdict or '—'} · SPD {spd_verdict or '—'}"]
        for r in races:
            parts.append(f"{_short(r)}: {_get(r, 'race', 'final_time') or '—'} · "
                         f"DET {_get(r, 'det', 'verdict') or '—'} · SPD {_get(r, 'spd', 'verdict') or '—'}")
        return "  |  ".join(parts)

    _race_section("detspd", "DET & SPD", "🔍 Compare DET & SPD with past races", build,
                  chips_fn=chips)


# ----------------------------------------------------------------------
# 4) EF
# ----------------------------------------------------------------------
def render_ef(df, ef_1, ef_2, ef_mean, ef_delta, verdict, block_min):
    def build(races):
        return [
            metric("Δ EF", _f(ef_delta), _past(races, "ef", "delta"), "zero", 2,
                   fmt=lambda v: _num(v, 2, signed=True)),
            metric("EF mean", _f(ef_mean), _past(races, "ef", "race_mean"), "up", 2),
            metric("EF 1st half", _f(ef_1), _past(races, "ef", "first_half"), "up", 2),
            metric("EF 2nd half", _f(ef_2), _past(races, "ef", "second_half"), "up", 2),
        ]

    def block_warnings(races):
        for r in races:
            pb = _get(r, "ef", "block_min")
            if pb is not None and pb != block_min:
                st.warning(f"⚠️ **{_label(r)}**: EF computed on {pb}-min blocks "
                           f"(this race: {block_min} min). Values are not strictly comparable.")

    def chips(races):
        parts = [f"This race: {duration_str(df)} · {verdict or '—'}"]
        for r in races:
            parts.append(f"{_short(r)}: {_get(r, 'race', 'final_time') or '—'} · "
                         f"{_get(r, 'ef', 'verdict') or '—'}")
        return "  |  ".join(parts)

    _race_section("ef", "Efficiency Factor", "🔍 Compare Efficiency Factor with past races",
                  build, chips_fn=chips, extra=block_warnings)


# ======================================================================
# 7) TIME IN ZONE — strip per zona (colore scelto dal coach) + heatmap
# ======================================================================
TIZ_DIR = {"⚪ No colour": "none", "🟢 More is better": "up", "🟢 Less is better": "down"}


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
        past = []                                     # (race, minuti, %)
        for r in races:
            mins = _tiz_minutes_past(r)
            if not mins or not sum(mins):
                st.caption(f"{_label(r)}: no time-in-zone data in the curriculum.")
                continue
            past.append((r, mins, [m / sum(mins) * 100 for m in mins]))

        chosen = _params("tiz", ZONES)
        st.markdown("**Colour for each zone** — what counts as better in this race:")
        dirs = {}
        for col, z in zip(st.columns(5), ZONES):
            dirs[z] = TIZ_DIR[col.selectbox(z, list(TIZ_DIR), key=f"cmp_tiz_dir_{z}")]
        trend, numbers = _options("tiz")

        metrics = []
        for i, z in enumerate(ZONES):
            if z not in chosen:
                continue
            metrics.append(metric(
                f"{z} (% of time)", cur_pct[i],
                [(_short(r), _get(r, "race", "date"), p[i]) for r, _, p in past],
                dirs[z], 0, unit=" pts", fmt=lambda v: f"{v:.0f}%",
                tol=TOL["zone"] if dirs[z] != "none" else None))
        blocks = []
        if metrics:
            chips = "  |  ".join(
                [f"This race: {duration_str(df)} · in zones {_hmm(sum(cur_min))}"]
                + [f"{_short(r)}: {_get(r, 'race', 'final_time') or '—'} · in zones {_hmm(sum(m))}"
                   for r, m, _ in past])
            blocks.append(_block(metrics, trend, numbers, [_short(r) for r, _, _ in past],
                                 chips=chips))

        # --- Heatmap (tutte le zone) ---
        cols = [("This race", duration_str(df), cur_min, cur_pct, True)]
        cols += [(_short(r), _get(r, "race", "final_time") or "—", m, p, False) for r, m, p in past]
        x = [f"{c[0]}<br>race {c[1]}" for c in cols]
        z = np.array([c[3] for c in cols]).T
        text = [[f"{p:.0f}%<br>{_hmm(m)}" for p, m in zip(c[3], c[2])] for c in cols]
        fig = go.Figure(go.Heatmap(
            z=z, x=x, y=ZONES, text=np.array(text).T, texttemplate="%{text}",
            colorscale="YlOrRd", zmin=0, colorbar=dict(title="% time"),
            hovertemplate="%{y}: %{z:.0f}%<extra></extra>",
        ))
        fig.update_layout(title="Time in zone — % of race time and total time (h:mm) per zone",
                          yaxis=dict(autorange="reversed"), height=400,
                          margin=dict(t=50, b=60, l=50, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Total race only, in % (races of different length can't be compared in minutes).")
        _pdf_toggle("tiz", "Time in zone", blocks, races=[r for r, _, _ in past],
                    heatmap={"cols": [f"{c[0]}\nrace {c[1]}" for c in cols],
                             "z": z.tolist(), "zones": ZONES,
                             "hmm": np.array([[_hmm(m) for m in c[2]] for c in cols]).T.tolist()})


# ======================================================================
# 5) CLIMBS — un blocco per salita di questa gara
# ======================================================================
def _climb_info(row, pos=None):
    grade, vam, hr = _f(row.get("Avg Grade (%)")), _f(row.get("VAM (m/h)")), _f(row.get("Avg FC"))
    pcts = [row.get(f"% {z}") for z in ZONES]
    if pos is None:
        pos = _f(row.get("Position (%)"))
    return {
        "name": row.get("Climb Name", "?"), "grade": grade, "vam": vam, "hr": hr,
        "length": _f(row.get("Distance (km)")), "duration": _hms_to_sec(row.get("Duration")),
        "gain": _f(row.get("Elevation (m)")),
        "efs": _f(row.get("Avg EFS (km/h)")),
        "vam_bpm": vam / hr if vam is not None and hr else None,
        "dominant": dominant_zone(pcts), "pos": pos,
    }


def _similar_by(c, p):
    """'same climb', 'grade', 'length', 'grade + length' o None."""
    if not CLIMB_AUTO_NAME_RE.match(str(c["name"])) and _slug(p["name"]) == _slug(c["name"]):
        return "same climb"
    by = []
    if c["grade"] is not None and p["grade"] is not None and abs(p["grade"] - c["grade"]) <= GRADE_WINDOW:
        by.append("grade")
    if c["length"] and p["length"] is not None and abs(p["length"] - c["length"]) / c["length"] <= LENGTH_WINDOW:
        by.append("length")
    return " + ".join(by) or None


def _hms(s):
    if s is None:
        return "—"
    s = int(s)
    return f"{s // 3600}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# Strip: solo prestazione. Lunghezza, durata, pendenza, D+ e posizione stanno
# nella tabella sopra (servono a decidere se la salita è confrontabile).
CLIMB_PARAMS = ["VAM (m/h)", "VAM/bpm", "EFS (km/h)", "Avg HR (bpm)"]


def _climb_metrics(c, cands, chosen):
    def past(key):
        return [(f"{_short(r)} · {p['name']}", _get(r, "race", "date"), p[key]) for r, p, _ in cands]
    spec = {
        "VAM (m/h)": ("vam", "up", 0, None),
        "VAM/bpm": ("vam_bpm", "up", 2, None),        # FC assoluta
        "EFS (km/h)": ("efs", "up", 2, None),
        "Avg HR (bpm)": ("hr", "none", 0, None),      # FC assoluta
    }
    out = []
    for label in CLIMB_PARAMS:
        if label not in chosen:
            continue
        key, d, nd, fmt = spec[label]
        out.append(metric(label, c[key], past(key), d, nd, fmt=fmt,
                          hr_abs=label in ("VAM/bpm", "Avg HR (bpm)")))
    return out


def _climb_context_table(c, cands, included):
    """Tabella da colpo d'occhio: questa salita + candidate, dati del percorso."""
    def row(race, p, by, inc):
        return {"✓": inc, "Race": race, "Climb": p["name"], "Similar by": by,
                "Length (km)": _num(p["length"], 1), "Duration": _hms(p["duration"]),
                "Grade (%)": _num(p["grade"], 0), "D+ (m)": _num(p["gain"], 0),
                "Position": f"{p['pos']:.0f}%" if p["pos"] is not None else "—",
                "Main zone": p["dominant"]}
    rows = [row("▶ This race", c, "—", "")]
    rows += [row(_short(r), p, by, "✓" if (r, p, by) in included else "✗") for r, p, by in cands]
    return pd.DataFrame(rows)


def render_climbs(climb_df, climb_data, df):
    if not _past_races() or climb_df is None or climb_df.empty:
        return
    with st.expander("🔍 Compare climbs with past races"):
        races = _pick("climbs")
        if not races:
            return
        pos = climb_positions(climb_data, df)
        current = [_climb_info(row, pos.get(row.get("Climb Name"), (None, None))[1])
                   for row in climb_df.to_dict(orient="records")]
        st.markdown("**Climbs of this race to compare:**")
        current = _checkbox_grid([(c["name"], c) for c in current], "cmp_climb_sel", per_row=6)
        if not current:
            st.info("Select at least one climb.")
            return
        chosen = _params("climbs", CLIMB_PARAMS)
        trend, numbers = _options("climbs")
        st.caption(f"Compared with: the same climb in other races (names typed by hand), "
                   f"or climbs with grade ±{GRADE_WINDOW} points or length "
                   f"±{LENGTH_WINDOW * 100:.0f}%.")
        blocks = []
        for c in current:
            cands = []
            for r in races:
                for row in r.get("climbs") or []:
                    p = _climb_info(row)
                    by = _similar_by(c, p)
                    if by:
                        cands.append((r, p, by))
            title = (f"⛰️ {c['name']} — length: {_num(c['length'], 1)} km, "
                     f"grade: {_num(c['grade'], 0)}%, main zone {c['dominant']}")
            st.markdown(f"#### {title}")
            if not cands:
                st.caption("No similar climb in the selected races.")
                continue
            # 1) scelta delle salite, 2) tabella del percorso, 3) strip prestazione
            st.markdown("**Climbs to include in the comparison:**")
            kept = _checkbox_grid(
                [(f"{_short(r)} · {p['name']} ({by})", (r, p, by)) for r, p, by in cands],
                f"cmp_cl_{_slug(c['name'])}")
            context = _climb_context_table(c, cands, kept)
            st.dataframe(context, hide_index=True, use_container_width=True)
            if not kept:
                st.caption("All climbs unticked.")
                continue
            metrics = _climb_metrics(c, kept, chosen)
            if not metrics:
                st.info("Choose at least one parameter.")
                return
            blocks.append(_block(metrics, trend, numbers,
                                 [f"{_short(r)} · {p['name']}" for r, p, _ in kept],
                                 subtitle=title.replace("⛰️ ", ""),
                                 context=context[context["✓"] != "✗"].drop(columns="✓")))
        if blocks:
            _pdf_toggle("climbs", "Climbs", blocks, races=races)


# ======================================================================
# 6) LAPS — Lap 1 contro Lap 1, Lap 2 contro Lap 2, ...
# ======================================================================
LAP_PARAMS = ["Speed (km/h)", "Avg EFS (km/h)", "Pace (min/km)", "Avg HR (bpm)",
              "Distance (km)", "Elevation (m)", "Duration"]
_LAP_SPEC = {
    "Speed (km/h)": ("Speed (km/h)", "up", 1, _f, None),
    "Avg EFS (km/h)": ("Avg EFS (km/h)", "up", 2, _f, None),
    "Pace (min/km)": ("Lap Pace (min/km)", "down", 0, _pace_to_sec, _sec_to_pace),
    "Avg HR (bpm)": ("Avg FC", "none", 0, _f, None),
    "Distance (km)": ("Distance (km)", "context", 1, _f, None),
    "Elevation (m)": ("Elevation (m)", "context", 0, _f, None),
    "Duration": ("Duration", "context", 0, _hms_to_sec, _hms),
}


def render_laps(lap_df):
    if not _past_races() or lap_df is None or lap_df.empty:
        return
    with st.expander("🔍 Compare laps with past races"):
        races = _pick("laps")
        if not races:
            return
        cur_laps = lap_df.to_dict(orient="records")
        past = [(r, r.get("laps") or []) for r in races]
        n_max = max([len(cur_laps)] + [len(l) for _, l in past])
        st.markdown("**Laps to compare:**")
        chosen_laps = _checkbox_grid([(f"Lap {n}", n) for n in range(1, n_max + 1)],
                                     "cmp_lap_sel", per_row=6)
        if not chosen_laps:
            st.info("Select at least one lap.")
            return
        chosen = _params("laps", LAP_PARAMS)
        trend, numbers = _options("laps")
        blocks = []
        for n in chosen_laps:
            cur = cur_laps[n - 1] if n <= len(cur_laps) else None
            others = [(r, laps[n - 1]) for r, laps in past if n <= len(laps)]
            st.markdown(f"#### 🏃 Lap {n}")
            if cur is None or not others:
                st.caption("This lap number doesn't exist in this race or in the selected races.")
                continue
            metrics = []
            for label in LAP_PARAMS:
                if label not in chosen:
                    continue
                col, d, nd, conv, fmt = _LAP_SPEC[label]
                tol = TOL["Avg EFS (km/h) lap"] if label == "Avg EFS (km/h)" else None
                metrics.append(metric(label, conv(cur.get(col)),
                                      [(_short(r), _get(r, "race", "date"), conv(l.get(col)))
                                       for r, l in others], d, nd, fmt=fmt,
                                      hr_abs=label == "Avg HR (bpm)", tol=tol))
            if not metrics:
                st.info("Choose at least one parameter.")
                return
            blocks.append(_block(metrics, trend, numbers, [_short(r) for r, _ in others],
                                 subtitle=f"Lap {n}"))
        if blocks:
            _pdf_toggle("laps", "Laps", blocks, races=races)