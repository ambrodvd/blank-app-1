"""4) DET index, grafico live FC/EFS/quota, Speed Decadence."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression


# ======================================================================
# HELPERS
# ======================================================================
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


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "X", "comment", "det_index_str", "df_clean", "efs_df", "efs_totals", "reg",
    "spd_index", "spd_kmh_per_h", "spd_comment", "_hr_bpm_per_h",
)


def render(ctx):
    analysis_ready = ctx.get("analysis_ready")
    df = ctx.get("df")

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

    if analysis_ready:
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
        # EFFICIENCY FACTOR — velocità per sforzo relativo
        # =====================================================================
        # EF = EFS / (FC / soglia): la velocità equivalente pianeggiante che
        # l'atleta terrebbe se corresse esattamente a soglia. La FC è
        # normalizzata sulla soglia perché la FC assoluta è in larga parte
        # genetica e non dice nulla su velocità o fitness: dividendo per z4,
        # due atleti alla stessa intensità relativa ottengono lo stesso EF.
        #

            st.divider()
            st.markdown("### Efficiency Factor — velocità per sforzo relativo")

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})