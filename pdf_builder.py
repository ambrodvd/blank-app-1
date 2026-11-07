"""9) Report PDF: classe PDF, grafici matplotlib e generazione."""
import os
import tempfile

import streamlit as st
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.linear_model import LinearRegression

from comparison import DOT_COLOR, cur_kind
from efficiency import ef_delta_verdict
from input_data import SEGMENT_COLORS, format_hmm, hhmm_axis_formatter, hhmm_to_seconds


# ======================================================================
# HELPERS
# ======================================================================
LOGO_DU_PATH   = "LOGO_DU_BIANCO.jpeg"
SCRITTA_PATH   = "SCRITTA_ULTRANERD.png"


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


def add_chart_to_pdf(pdf, fig, title=None):
    if title:
        pdf.section_title(title)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        fig.savefig(tmpfile.name, format="PNG", dpi=200, bbox_inches="tight")
        tmpfile.close()
        pdf.image(tmpfile.name, x=10, w=190)
    plt.close(fig)


# --- Reusable helper: elevation profile chart for PDF ---
def pdf_elevation_with_segments(df, data, mode, title):
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
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(hhmm_axis_formatter))
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


# --- Confronti con il curriculum (in fondo al PDF) ---
# I font standard di FPDF sono latin-1: niente Δ, ▶, ⚠️, trattini lunghi.
_PDF_CHARS = {"(Δ ": "(", "Δ": "Delta", "▶ ": "> ", "—": "-", "–": "-", "⚠️": "!", "≤": "<="}


def _pdf_safe(text):
    text = str(text)
    for k, v in _PDF_CHARS.items():
        text = text.replace(k, v)
    return text.encode("latin-1", "replace").decode("latin-1")


def _fit(pdf, text, width):
    """Tronca il testo perché stia nella cella (con un margine)."""
    while text and pdf.get_string_width(text) > width - 1.5:
        text = text[:-1]
    return text


def pdf_write_comparison_table(pdf, table, subtitle=""):
    """Tabella generica: larghezze proporzionali al contenuto, font piccolo."""
    if table is None or table.empty:
        return
    cols = [str(c) for c in table.columns]
    data = [[_pdf_safe(v) for v in row] for row in table.astype(str).values.tolist()]
    heads = [_pdf_safe(c) for c in cols]
    size = 7 if len(cols) <= 10 else 6
    pdf.set_font("Helvetica", "", size)
    need = [max([pdf.get_string_width(h[:16])] + [pdf.get_string_width(r[i]) for r in data]) + 3
            for i, h in enumerate(heads)]
    avail = pdf.w - pdf.l_margin - pdf.r_margin
    widths = [w * avail / sum(need) for w in need]
    row_h = 5

    if pdf.get_y() + row_h * (len(data) + 2) > pdf.h - pdf.b_margin and len(data) < 30:
        pdf.add_page()                        # la tabella resta intera su una pagina
    if subtitle:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(40, 40, 40)
        pdf.cell(0, 6, _pdf_safe(subtitle), ln=True)
    pdf.set_font("Helvetica", "B", size)
    pdf.set_fill_color(245, 245, 245)
    pdf.set_text_color(20, 20, 20)
    for h, w in zip(heads, widths):
        pdf.cell(w, row_h, _fit(pdf, h, w), border=1, fill=True, align="C")
    pdf.ln(row_h)
    for r_i, row in enumerate(data):
        current = any(v.startswith("> This race") for v in row)   # in grassetto
        pdf.set_font("Helvetica", "B" if current else "", size)
        pdf.set_text_color(20, 20, 20) if current else pdf.set_text_color(60, 60, 60)
        for v, w in zip(row, widths):
            pdf.cell(w, row_h, _fit(pdf, v, w), border=1, align="C")
        pdf.ln(row_h)
    pdf.add_spacer(4)


# Stessi colori e stessa logica del pallino dell'app (comparison.py)
_PILL_COLORS = {**DOT_COLOR, "good_light": "#4caf50", "bad_light": "#e57373",
                "neutral": "#2a78d6"}          # blu come il pallino di questa gara
_DOT_COLORS = DOT_COLOR
_cur_kind = cur_kind


def pdf_strip_chart(rows):
    """Righe parametro come nell'app: etichetta, valore, dot strip, verdetti."""
    heights = [max(1, len(r["pills"])) for r in rows]
    unit = 0.30                                   # pollici per riga di verdetto
    fig_h = sum(heights) * unit + 0.4
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, sum(heights))
    ax.axis("off")
    top = sum(heights)
    for r, h in zip(rows, heights):
        mid = top - h / 2
        ctx = r["direction"] == "context"
        ax.text(0, mid, r["label"], va="center", fontsize=9,
                style="italic" if ctx else "normal")
        ax.text(2.9, mid, r["cur_fmt"], va="center", ha="right", fontsize=11, weight="bold",
                style="italic" if ctx else "normal")
        vals = [v for _, _, v, _ in r["past"] if v is not None] + ([r["cur"]] if r["cur"] is not None else [])
        if vals:
            lo, hi = min(vals), max(vals)
            pad = (hi - lo) * 0.08 or (abs(hi) * 0.05 or 1)
            lo, hi = lo - pad, hi + pad
            xp = lambda v: 3.2 + (v - lo) / (hi - lo) * 3.4
            ax.plot([3.15, 6.65], [mid, mid], color="#bbbbbb", lw=0.8, zorder=1)
            for k, (_, _, v, txt) in enumerate(r["past"]):
                if v is not None:
                    ax.scatter([xp(v)], [mid], s=30, color="#888780", zorder=2)
                    # valore sopra/sotto, alternati per non sovrapporsi
                    ax.text(xp(v), mid + (0.3 if k % 2 == 0 else -0.3), txt, ha="center",
                            va="center", fontsize=6.5, color="#666666")
            if r["cur"] is not None:
                ax.scatter([xp(r["cur"])], [mid], s=80, color=_DOT_COLORS[_cur_kind(r["pills"])],
                           edgecolor="white",
                           linewidth=1.2, zorder=3)
        for j, (txt, kind) in enumerate(r["pills"]):
            ax.text(6.9, top - (j + 0.5), txt, va="center", fontsize=7.5, color=_PILL_COLORS[kind])
        ax.plot([0, 10], [top - h, top - h], color="#e5e5e5", lw=0.5)
        top -= h
    plt.tight_layout(pad=0.2)
    return fig


def pdf_trend_chart(rows, this_date):
    """Trend nel tempo, un piccolo grafico per parametro (3 per riga)."""
    rows = [r for r in rows if sum(v is not None for _, _, v, _ in r["past"]) + 1 >= 2]
    if not rows:
        return None
    ncols = min(3, len(rows))
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 2.1 * nrows), squeeze=False)
    for ax in axes.flat[len(rows):]:
        ax.axis("off")
    for ax, r in zip(axes.flat, rows):
        pts = [(d, v, False) for _, d, v, _ in r["past"] if v is not None and d]
        if r["cur"] is not None and this_date and this_date != "—":
            pts.append((this_date, r["cur"], True))
        pts.sort(key=lambda p: p[0])
        x = pd.to_datetime([p[0] for p in pts])
        y = [p[1] for p in pts]
        ax.plot(x, y, color="gray", marker="o", lw=1.2, ms=4)
        for xi, yi, p in zip(x, y, pts):
            if p[2]:
                ax.scatter([xi], [yi], s=70, color=_DOT_COLORS[_cur_kind(r["pills"])],
                           zorder=3, edgecolor="white")
        ax.set_title(r["label"], fontsize=9)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        for lab in ax.get_xticklabels():
            lab.set_rotation(20)
    plt.tight_layout()
    return fig


def pdf_tiz_heatmap(data):
    """% tempo in zona: righe Z1-Z5, colonne = gare."""
    z = pd.DataFrame(data["z"], index=data["zones"],
                     columns=[_pdf_safe(c) for c in data["cols"]])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    # La % è già nelle righe sopra: nelle celle il tempo totale in zona (h:mm),
    # il colore resta la % del tempo gara.
    labels = np.array(data["hmm"]) if data.get("hmm") else True
    sns.heatmap(z, annot=labels, fmt="" if data.get("hmm") else ".0f", cmap="YlOrRd", vmin=0,
                linewidths=0.5, linecolor="white", cbar_kws={"label": "% of race time"}, ax=ax)
    ax.set_title("Total time in zone (h:mm) — colour = % of race time", fontsize=9)
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    plt.tight_layout()
    return fig


def pdf_write_comparisons(pdf, items):
    """I confronti spuntati con "Add this comparison to the PDF", come in app:
    parametri scelti (dot strip), trend e tabella solo se attivi, heatmap."""
    if not items:
        return
    pdf.add_page()
    pdf.section_title("Comparison with past races")
    pdf.set_font("Helvetica", "", 7)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(0, 3.5, "Big dot = this race: green = better than all selected races, red = worse "
                           "than all, amber = mixed, blue = same (within the tolerance band) or no "
                           "better/worse direction. "
                           "Grey dots = selected races. Delta = this race minus the selected race.")
    pdf.ln(2)
    for item in items.values():
        if pdf.get_y() > pdf.h - 70:              # niente titolo orfano a fondo pagina
            pdf.add_page()
        pdf.section_title(_pdf_safe(item["title"]))
        if item.get("races"):
            # gare usate nel confronto, evidenziate
            pdf.set_fill_color(230, 240, 251)
            pdf.set_text_color(12, 68, 124)
            pdf.set_font("Helvetica", "B", 9)
            pdf.multi_cell(0, 5.5, _pdf_safe("Compared with: " + "  |  ".join(item["races"])),
                           fill=True)
            pdf.ln(2)
        for block in item.get("blocks") or []:
            if pdf.get_y() > pdf.h - 60:
                pdf.add_page()
            if block.get("subtitle"):
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_text_color(40, 40, 40)
                pdf.cell(0, 6, _pdf_safe(block["subtitle"]), ln=True)
            if block.get("chips"):
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(90, 90, 90)
                pdf.multi_cell(0, 4, _pdf_safe(block["chips"]))
                pdf.ln(1)
            if block.get("context") is not None:
                pdf_write_comparison_table(pdf, block["context"])
            add_chart_to_pdf(pdf, pdf_strip_chart(block["rows"]))
            if block.get("trend"):
                fig = pdf_trend_chart(block["rows"], block.get("this_date"))
                if fig is not None:
                    add_chart_to_pdf(pdf, fig)
            if block.get("table") is not None:
                pdf_write_comparison_table(pdf, block["table"])
        if item.get("heatmap"):
            add_chart_to_pdf(pdf, pdf_tiz_heatmap(item["heatmap"]))


# --- Reusable helper: write a lap/climb info + HR table to PDF ---
def pdf_write_analysis_table(pdf, zone_df, section_label, name_col):
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

    # HR zone columns table: solo le colonne Z / % Z presenti (selezionate)
    zone_cols = [c for c in zone_df_copy.columns if c.startswith("Z") or c.startswith("%")]
    if not zone_cols:
        return
    hr_cols   = [name_col] + zone_cols
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
        for col in hr_cols:
            pdf.cell(col_width, row_height, str(row.get(col, "")), border=1, align='C')
        pdf.ln(row_height)
    pdf.add_spacer(6)


# ======================================================================
# RENDER
# ======================================================================
def render(ctx):
    EF_BLOCK_MIN = ctx.get("EF_BLOCK_MIN")
    X = ctx.get("X")
    _ctr = ctx.get("_ctr")
    _ef = ctx.get("_ef")
    _ef_efs = ctx.get("_ef_efs")
    _ef_rel = ctx.get("_ef_rel")
    _hr_thr = ctx.get("_hr_thr")
    analysis_ready = ctx.get("analysis_ready")
    athlete_name = ctx.get("athlete_name")
    bar_df = ctx.get("bar_df")
    combined_df = ctx.get("combined_df")
    comment = ctx.get("comment")
    det_index_str = ctx.get("det_index_str")
    df = ctx.get("df")
    df_clean = ctx.get("df_clean")
    efs_df = ctx.get("efs_df")
    efs_totals = ctx.get("efs_totals")
    final_time_str = ctx.get("final_time_str")
    first_half_avg = ctx.get("first_half_avg")
    formatted_date = ctx.get("formatted_date")
    heatmap_df_minutes = ctx.get("heatmap_df_minutes")
    kilometers = ctx.get("kilometers")
    overall_avg = ctx.get("overall_avg")
    percent_diff = ctx.get("percent_diff")
    race_name = ctx.get("race_name")
    reg = ctx.get("reg")
    second_half_avg = ctx.get("second_half_avg")
    spd_index = ctx.get("spd_index")
    spd_kmh_per_h = ctx.get("spd_kmh_per_h")
    total_elevation_gain = ctx.get("total_elevation_gain")
    z1 = ctx.get("z1")
    z2 = ctx.get("z2")
    z3 = ctx.get("z3")
    z4 = ctx.get("z4")
    z5 = ctx.get("z5")

    if analysis_ready and df is not None and not df.empty and 'HR Zone' in df.columns and 'athlete_name' in st.session_state:
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
            if df is not None and not df.empty:
                fig2, ax2 = plt.subplots(figsize=(10, 4))

                ax2.plot(df["elapsed_sec"], df["elevation_m"], color="gray", label="Elevation")

                segment_colors = SEGMENT_COLORS

                for i in range(1, st.session_state.get('num_segments', 1) + 1):
                    start_key = f'segment{i}_start'
                    end_key   = f'segment{i}_end'
                    if all(k in st.session_state for k in [start_key, end_key]):
                        seg_start = hhmm_to_seconds(st.session_state[start_key])
                        seg_end   = hhmm_to_seconds(st.session_state[end_key])
                        if seg_start is not None and seg_end is not None and seg_end > seg_start:
                            mask      = (df["elapsed_sec"] >= seg_start) & (df["elapsed_sec"] <= seg_end)
                            seg_label = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                            ax2.fill_between(
                                df["elapsed_sec"], df["elevation_m"],
                                where=mask, alpha=0.25,
                                color=segment_colors[(i-1) % len(segment_colors)],
                                label=seg_label
                            )

                ax2.xaxis.set_major_formatter(mticker.FuncFormatter(hhmm_axis_formatter))
                ax2.xaxis.set_major_locator(mticker.MaxNLocator(nbins=10))
                plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
                ax2.set_xlabel("Elapsed Time (hh:mm)")
                ax2.set_ylabel("Elevation (m)")
                ax2.set_title("Elevation Profile with Time Segments Highlighted")
                ax2.grid(True)
                ax2.legend(loc="upper left", fontsize=8)
                plt.tight_layout()
                add_chart_to_pdf(pdf, fig2, title="Elevation Profile - Time Segments")

            # --- Time-in-Zone Table ---
            if combined_df is not None:
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
            if bar_df is not None:
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
                add_chart_to_pdf(pdf, fig, title="Time-in-Zone - Bar Chart")
                pdf.add_page()

            if heatmap_df_minutes is not None and not heatmap_df_minutes.empty:
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
                add_chart_to_pdf(pdf, fig, title="Time-in-Zone - Heatmap")
                pdf.add_page()

            # ===========================================================
            # ⛰️ CLIMB ANALYSIS IN PDF
            # ===========================================================
            if st.session_state.get("do_climb_analysis") and st.session_state.get("climb_data"):
                climb_data   = st.session_state["climb_data"]
                climb_zone_df = st.session_state.get("climb_zone_df")

                fig = pdf_elevation_with_segments(df, climb_data, mode="climb", title="Elevation Profile - Climbs Highlighted")
                add_chart_to_pdf(pdf, fig, title="Climb Analysis")

                if climb_zone_df is not None and not climb_zone_df.empty:
                    # Solo le colonne spuntate nelle checkbox "Data to show"
                    _vis = st.session_state.get("climb_visible_cols")
                    if _vis:
                        climb_zone_df = climb_zone_df[[c for c in _vis if c in climb_zone_df.columns]]
                    pdf_write_analysis_table(pdf, climb_zone_df, "Climb Analysis", "Climb Name")
                    if "% Z1" in climb_zone_df.columns:
                        fig = pdf_zone_bar_chart(climb_zone_df, "Climb Name", "Climb — % Time in HR Zones")
                        add_chart_to_pdf(pdf, fig)
                    pdf.add_page()

            # ===========================================================
            # 🏃 LAP ANALYSIS IN PDF
            # ===========================================================
            if st.session_state.get("do_lap_analysis") and st.session_state.get("lap_data"):
                lap_data    = st.session_state["lap_data"]
                lap_zone_df = st.session_state.get("lap_zone_df")

                fig = pdf_elevation_with_segments(df, lap_data, mode="lap", title="Elevation Profile - Laps Highlighted")
                add_chart_to_pdf(pdf, fig, title="Lap Analysis")

                if lap_zone_df is not None and not lap_zone_df.empty:
                    # Solo le colonne spuntate nelle checkbox "Data to show"
                    _vis = st.session_state.get("lap_visible_cols")
                    if _vis:
                        lap_zone_df = lap_zone_df[[c for c in _vis if c in lap_zone_df.columns]]
                    pdf_write_analysis_table(pdf, lap_zone_df, "Lap Analysis", "Lap Name")
                    if "% Z1" in lap_zone_df.columns:
                        fig = pdf_zone_bar_chart(lap_zone_df, "Lap Name", "Lap — % Time in HR Zones")
                        add_chart_to_pdf(pdf, fig)
                    pdf.add_page()

            # --- HR + EFS + Elevation Trend Chart ---
            # Tre bande impilate con asse x condiviso, stessa lettura del
            # grafico live. Le proporzioni ricalcano i pesi della UI
            # (42/28/30). efs_df arriva dalla sezione live, già calcolato.
            _efs_ok = (
                efs_df is not None
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
            add_chart_to_pdf(pdf, fig, title="Trend Analysis - HR, EFS & Elevation")

            # Stessi due indici della sezione live: DET sulla FC, SPD sulla
            # velocità, ciascuno con il tasso grezzo accanto.
            _hr_rate_pdf = float(reg.coef_[0]) * 3600.0
            pdf.body_text(
                f"DET Index: {det_index_str} ({comment})   |   "
                f"{_hr_rate_pdf:+.3f} bpm per hour"
            )
            if spd_index is not None and spd_index is not None:
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
            if efs_totals is not None and efs_totals is not None:
                pdf.body_text(
                    f"Total EFD: {efs_totals['efd_km']:.2f} km  |  "
                    f"Average EFS: {efs_totals['avg_efs_kmh']:.2f} km/h"
                )
            pdf.add_spacer(4)

                    # --- Efficiency Factor ---
            # _ef, _ef_x, _ef_rel, _ef_efs, _hr_thr arrivano dalla sezione live.
            # Se l'analisi EF non è stata eseguita (traccia corta, soglia non
            # impostata) il blocco si salta senza rompere il report.
            # _ctr (centri dei blocchi) arriva dalla sezione live insieme a _ef:
            # va in guardia anche lui, altrimenti un rerun in cui l'analisi EF
            # non è stata eseguita lascia il nome non definito.
            if (_ef is not None and _ef is not None
                    and _ctr is not None and len(_ctr) > 3):
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

                ax_ef.plot(_ctr, _ef, color="#e07b39", linewidth=2.2,
                           label="EF (km/h a soglia)", zorder=3)
                ax_ef.set_ylabel("EF (km/h per sforzo relativo)", fontsize=9)
                ax_ef.set_xlabel("Elapsed Time (hours)", fontsize=9)
                ax_ef.grid(True, alpha=0.3)
                ax_ef.tick_params(labelsize=8)
                ax_ef.set_zorder(ax_bg.get_zorder() + 1)
                ax_ef.patch.set_visible(False)

                ax_hrp = ax_ef.twinx()
                ax_hrp.plot(_ctr, _ef_rel * 100.0, color="royalblue",
                            linewidth=1.1, alpha=0.7, label="FC (% soglia)")
                ax_hrp.set_ylabel("FC (% soglia)", fontsize=9)
                ax_hrp.tick_params(labelsize=8)

                # Terzo asse spostato in fuori: senza offset l'etichetta finisce
                # sopra quella della FC e diventano illeggibili entrambe.
                ax_efsp = ax_ef.twinx()
                ax_efsp.spines["right"].set_position(("outward", 42))
                ax_efsp.plot(_ctr, _ef_efs, color="seagreen",
                             linewidth=1.1, alpha=0.7, label="EFS (km/h)")
                ax_efsp.set_ylabel("EFS (km/h)", fontsize=9)
                ax_efsp.tick_params(labelsize=8)

                _h1, _l1 = ax_ef.get_legend_handles_labels()
                _h2, _l2 = ax_hrp.get_legend_handles_labels()
                _h3, _l3 = ax_efsp.get_legend_handles_labels()
                ax_ef.legend(_h1 + _h2 + _h3, _l1 + _l2 + _l3,
                             loc="upper right", fontsize=7, ncol=3)
                ax_ef.set_title(
                    f"Efficiency Factor (blocchi {EF_BLOCK_MIN} min) — "
                    "FC relativa alla soglia",
                    fontsize=11, fontweight="bold")

                plt.tight_layout()
                add_chart_to_pdf(pdf, fig_ефp, title="Efficiency Factor")

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
                    f"EF = EFS / (HR / threshold), threshold {_hr_thr:.0f} bpm, "
                    f"computed on consecutive {EF_BLOCK_MIN}-minute blocks. "
                    "Equivalent flat speed the athlete holds per unit of relative "
                    "effort; comparable across athletes with different thresholds."
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
                add_chart_to_pdf(pdf, fig, title="Heart Rate Density Distribution")

            for i in range(1, st.session_state.get('num_segments', 1) + 1):
                start_key = f'segment{i}_start'
                end_key   = f'segment{i}_end'
                if all(k in st.session_state for k in [start_key, end_key]):
                    _pdf_start_sec = hhmm_to_seconds(st.session_state[start_key])
                    _pdf_end_sec   = hhmm_to_seconds(st.session_state[end_key])
                    if _pdf_start_sec is not None and _pdf_end_sec is not None and _pdf_end_sec > _pdf_start_sec:
                        seg_name = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                        df_seg   = df[(df["elapsed_sec"] >= _pdf_start_sec) & (df["elapsed_sec"] <= _pdf_end_sec)]
                        hr_seg   = df_seg["heart_rate"].dropna()
                        if len(hr_seg) > 1:
                            fig = build_density_chart_matplotlib(
                                hr_seg, f"HR Density - {seg_name}", overall_avg, z1, z2, z3, z4, z5,
                                x_min=x_min_total, x_max=x_max_total
                            )
                            add_chart_to_pdf(pdf, fig)

            # --- Coach Comment ---
            if st.session_state.comment:
                pdf.section_title("Coach comment")
                pdf.body_text(f"{st.session_state.comment}")

            # --- Confronti con il curriculum (checkbox "Add to PDF") ---
            pdf_write_comparisons(pdf, st.session_state.get("cmp_pdf") or {})

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