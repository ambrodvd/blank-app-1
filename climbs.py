"""2b) Climb analysis: detector automatico, inserimento manuale, grafici."""
import re

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from input_data import hhmm_to_seconds, nearest_idx, seconds_to_hhmm


# ======================================================================
# HELPERS
# ======================================================================
# Climb naming & plotting helpers
# ---------------------------
# I nomi auto-generati vanno rinumerati quando l'utente cancella o riordina
# una riga: "Climb 1, Climb 3, Climb 4" è un artefatto dell'editor, non una
# scelta. I nomi scritti a mano ("Colle del Nivolet") NON si toccano.
CLIMB_AUTO_NAME_RE = re.compile(r"^\s*climb\s*\d+\s*$", re.IGNORECASE)


CLIMB_PALETTE = ["#e07b39", "#2a9d8f", "#4a7fd4", "#c0508f", "#8a63c9",
                 "#d1495b", "#3fa34d", "#c9a227", "#5c8a9e", "#a1683a"]


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


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "df",
)


def render(ctx):
    do_climb = ctx.get("do_climb")

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

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})
