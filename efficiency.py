"""5) Efficiency Factor."""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

import comparison
from det_graphs import compute_segment_efs
from input_data import seconds_to_hhmm


# ======================================================================
# HELPERS
# ======================================================================
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


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "EF_BLOCK_MIN", "_ctr", "_ef", "_ef_efs", "_ef_rel", "_hr_thr",
    "_ef_1", "_ef_2", "_ef_delta", "_ef_cmt",
)


def render(ctx):
    analysis_ready = ctx.get("analysis_ready")
    df = ctx.get("df")
    df_clean = ctx.get("df_clean")
    efs_df = ctx.get("efs_df")

    if analysis_ready:
        EF_MIN_BLOCK_FRAC = 0.5    # ultimo blocco: sotto questa frazione si scarta

        # Durata del blocco scelta dall'utente. Lo slider sta qui fuori dal
        # controllo su efs_df: un widget dentro un ramo condizionale sparisce
        # e ricompare al variare dei dati, e Streamlit ne perde lo stato.
        # I valori sono discreti e non continui perché fra 15 e 16 minuti non
        # cambia niente di leggibile, mentre fra 10 e 30 sì.
        EF_BLOCK_CHOICES = [5, 10, 15, 20, 30, 45, 60]
        EF_BLOCK_MIN = st.select_slider(
            "Ampiezza dei blocchi per l'Efficiency Factor (minuti)",
            options=EF_BLOCK_CHOICES,
            value=st.session_state.get("ef_block_min", 15),
            key="ef_block_min",
            help="Blocchi consecutivi e non sovrapposti. Blocchi corti mostrano "
                 "più dettaglio ma risentono della pendenza del singolo tratto; "
                 "blocchi lunghi isolano meglio la deriva da fatica.",
        )

        if efs_df is not None and efs_df["efs_kmh"].notna().any():

            _hr_h = df_clean["elapsed_hours"].to_numpy()
            _hr_v = df_clean["hr_smooth"].to_numpy()
            _ef_x, _ef_hr, _ef_efs, _ef_lab = [], [], [], []

            _blk = EF_BLOCK_MIN / 60.0
            _race_end_h = float(df["elapsed_sec"].iloc[-1]) / 3600.0
            _n_blocks = int(np.ceil(_race_end_h / _blk))

            for _k in range(_n_blocks):
                _t0, _t1 = _k * _blk, (_k + 1) * _blk
                # l'ultimo blocco è quasi sempre tronco: sotto metà durata la
                # media è su troppo pochi dati per stare accanto agli altri
                if (min(_t1, _race_end_h) - _t0) / _blk < EF_MIN_BLOCK_FRAC:
                    continue

                _mh = (_hr_h >= _t0) & (_hr_h < _t1)
                if _mh.sum() < 10:
                    continue
                _hrm = float(np.nanmean(_hr_v[_mh]))

                # EFS del blocco pesata sul TEMPO (EFD totale / tempo totale),
                # non media dei segmenti da 20 m: quella conta allo stesso modo
                # 20 m di salita e 20 m di discesa e gonfia la velocità.
                _seg = df[(df["elapsed_sec"] >= _t0 * 3600.0)
                          & (df["elapsed_sec"] < _t1 * 3600.0)]
                _efm = compute_segment_efs(_seg)

                if _hrm > 0 and np.isfinite(_efm):
                    _ef_x.append(_t0)
                    _ef_hr.append(_hrm)
                    _ef_efs.append(_efm)
                    _ef_lab.append(
                        f"{seconds_to_hhmm(_t0 * 3600)}–"
                        f"{seconds_to_hhmm(min(_t1, _race_end_h) * 3600)}")

            if len(_ef_x) < 4:
                st.info("Traccia troppo corta per l'analisi a blocchi.")
            else:
                _ef_x = np.array(_ef_x)
                _ef_hr = np.array(_ef_hr)
                _ef_efs = np.array(_ef_efs)

                _hr_thr = float(st.session_state.get("z4", 0) or 0)
                _ef_rel = _ef_hr / _hr_thr if _hr_thr > 0 else None
                _ef = _ef_efs / _ef_rel if _ef_rel is not None else None

                if _ef is None:
                    st.warning(
                        "⚠️ Imposta la FC di soglia (Zona 4) nella sezione "
                        "**Athlete Heart Rate Zones** per calcolare l'Efficiency Factor."
                    )
                    st.stop()

                def _pad(arr, frac=0.10):
                    """Range con margine: con tre assi sovrapposti Plotly
                autoscala ciascuno per conto suo e taglia i picchi."""
                    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
                    span = max(hi - lo, 1e-9)
                    return [lo - frac * span, hi + frac * span]

                # Ogni marker va al CENTRO del blocco che rappresenta, non al suo
                # inizio: è lì che cade il punto medio dei dati su cui la media è
                # calcolata. I gradini erano fedeli ma illeggibili; marker più
                # segmento retto mostrano che le misure sono discrete senza
                # inventare una transizione fra un blocco e l'altro.
                # L'ultimo blocco può essere tronco, quindi il suo centro si
                # calcola sulla durata effettiva.
                _ctr = np.array([
                    (t0 + min(t0 + _blk, _race_end_h)) / 2.0 for t0 in _ef_x
                ])
                _py_ef  = _ef
                _py_hr  = _ef_rel * 100.0
                _py_efs = _ef_efs

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
                    x=_ctr, y=_py_ef, mode="lines",
                    name="EF (km/h a soglia)",
                    line=dict(color="#e07b39", width=2.4),
                    customdata=_ef_lab,
                    hovertemplate="%{customdata}<br>EF: %{y:.2f} km/h"
                                  "<extra></extra>",
                ))
                fig_ef.add_trace(go.Scatter(
                    x=_ctr, y=_py_hr, mode="lines",
                    name="FC (% soglia)", yaxis="y2",
                    line=dict(color="rgba(70,140,220,0.6)", width=1.3),
                    customdata=_ef_lab,
                    hovertemplate="%{customdata}<br>FC: %{y:.0f}% della soglia"
                                  "<extra></extra>",
                ))
                fig_ef.add_trace(go.Scatter(
                    x=_ctr, y=_py_efs, mode="lines",
                    name="EFS (km/h)", yaxis="y3",
                    line=dict(color="rgba(42,157,143,0.6)", width=1.3),
                    customdata=_ef_lab,
                    hovertemplate="%{customdata}<br>EFS: %{y:.2f} km/h"
                                  "<extra></extra>",
                ))
                fig_ef.update_layout(
                    title=f"Efficiency Factor (blocchi {EF_BLOCK_MIN} min) — "
                          "FC relativa alla soglia",
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

                # Blocchi non sovrapposti: le due metà sono davvero due metà
                # distinte di gara, non due insiemi di finestre che condividono
                # un'ora di dati come nella versione a finestra mobile.
                _half = len(_ef) // 2
                _ef_1, _ef_2 = _ef[:_half].mean(), _ef[_half:].mean()
                _ef_delta = _ef_2 - _ef_1
                _ef_cmt, _ef_col = ef_delta_verdict(_ef_delta)

                # Il delta ha lo stesso peso visivo degli altri tre valori: è la
                # variabile che conta più del livello assoluto di EF.
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
                    f"Scala del Δ EF: ≥ {EF_DELTA_P75:+.2f} bassa perdita · "
                    f"da {EF_DELTA_P25:+.2f} a {EF_DELTA_P75:+.2f} nella norma · "
                    f"< {EF_DELTA_P25:+.2f} significativa (p25/p75 dell'archivio gare)."
                )

                st.caption(
                    f"EF = EFS ÷ (FC / soglia)"
                )

                # --- Confronto EF con il curriculum ---
                comparison.render_ef(df, _ef_1, _ef_2, _ef.mean(), _ef_delta, _ef_cmt,
                                     EF_BLOCK_MIN)

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})
