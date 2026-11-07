"""8) Tabelle e grafici di analisi Lap / Climb."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from det_graphs import compute_segment_efs
import comparison
from climbs import _hex_rgba
from comparison import tom_low_score
from input_data import (
    CHART_PALETTE,
    format_hm,
    format_hms,
    parse_time_to_seconds,
    seconds_to_hhmm,
)


# ======================================================================
# RENDER
# ======================================================================
def render(ctx):
    df = ctx.get("df")

    if df is None or df.empty:
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

        colors = CHART_PALETTE

        # --- Helpers ---

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

                # .mean()/.max() su una fetta senza dati validi restituiscono NaN,
                # e int(NaN) solleva ValueError invece di dare 0.
                def _num(val, default=0.0):
                    try:
                        f = float(val)
                    except (TypeError, ValueError):
                        return default
                    return default if not np.isfinite(f) else f

                avg_fc    = int(_num(df_seg["heart_rate"].mean())) if "heart_rate" in df_seg.columns else 0
                distance  = round(_num(df_seg["distance_km"].max()) - _num(df_seg["distance_km"].min()), 1)
                elevation = int(_num(df_seg["elevation_m"].diff().clip(lower=0).sum()))

                if mode == "climb":
                    avg_grade = round((elevation / distance / 10) if distance > 0 else 0)
                    vam       = round(elevation / (duration_sec / 3600) if duration_sec > 0 else 0)
                    efs_climb = compute_segment_efs(df_seg)
                    efs_str   = f"{efs_climb:.2f}" if pd.notna(efs_climb) else "-"
                    climb_speed = f"{distance / (duration_sec / 3600):.1f}" if distance > 0 else "-"
                    rows.append([
                        entry.get("name", "Climb"),
                        duration_hms, distance, elevation,
                        avg_fc, avg_grade, vam, climb_speed, efs_str
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

                    # Velocità reale e EFS (velocità equivalente in piano, Minetti)
                    lap_speed = f"{distance / (duration_sec / 3600):.1f}" if distance > 0 else "-"
                    efs_lap   = compute_segment_efs(df_seg)
                    efs_str   = f"{efs_lap:.2f}" if pd.notna(efs_lap) else "-"

                    rows.append([
                        entry.get("name", "Lap"),
                        duration_hms, distance, elevation,
                        avg_fc, lap_pace, lap_speed, efs_str
                    ] + lap_summary_hm + pct_zones)

            if not rows:
                return None, None

            if mode == "climb":
                extra_cols = ["Avg FC", "Avg Grade (%)", "VAM (m/h)", "Speed (km/h)", "Avg EFS (km/h)"]
                name_col   = "Climb Name"
            else:
                extra_cols = ["Avg FC", "Lap Pace (min/km)", "Speed (km/h)", "Avg EFS (km/h)"]
                name_col   = "Lap Name"

            columns   = [name_col, "Duration", "Distance (km)", "Elevation (m)"] + extra_cols + zone_order + pct_cols
            result_df = pd.DataFrame(rows, columns=columns)
            result_df["Distance (km)"] = result_df["Distance (km)"].astype(float).map("{:.1f}".format)
            return result_df, name_col

        def pick_visible_columns(table, name_col, groups, key_prefix):
            """Riga di checkbox "Data to show" sopra una tabella Lap / Climb.
            Nome, durata, distanza e dislivello sono sempre visibili; ogni
            checkbox aggiunge il suo gruppo di colonne. La selezione vale
            anche per il PDF (session_state["<prefix>_visible_cols"])."""
            st.markdown("**Data to show:**")
            visible = [name_col, "Duration", "Distance (km)", "Elevation (m)"]
            for cb_col, (label, cols) in zip(st.columns(len(groups)), groups.items()):
                _k = f"{key_prefix}_show_{cols[0]}"
                if _k not in st.session_state:
                    st.session_state[_k] = True
                if cb_col.checkbox(label, key=_k):
                    visible += cols
            # Il PDF legge questa lista: stesse colonne della tabella a schermo
            st.session_state[f"{key_prefix}_visible_cols"] = visible
            return table[visible]

        # Stile assi comune: griglia e bordi recessivi, il dato in primo piano
        _AXIS = dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)",
                     showline=True, linecolor="rgba(128,128,128,0.55)",
                     mirror=True, ticks="outside", ticklen=4, zeroline=False)
        _LABEL_INK = "rgba(110,110,110,1)"

        def show_elevation_profile(data, mode):
            """Profilo altimetrico lisciato con ogni lap/salita evidenziato.

            - quota lisciata con media mobile centrata (niente scalini GPS/baro)
            - asse y stretto sulla quota reale, non da 0 m
            - lap: asse x in km; climb: asse x in ore (numerico, non stringhe)
            """
            ele  = df["elevation_m"].rolling(window=25, center=True, min_periods=1).mean()
            hhmm = df["elapsed_sec"].apply(seconds_to_hhmm)
            if mode == "lap":
                x       = df["distance_km"]
                x_label = "Distance (km)"
                x_hover = "%{x:.2f} km · %{customdata}"
            else:
                x       = df["elapsed_sec"] / 3600.0
                x_label = "Elapsed time (h)"
                x_hover = "%{customdata}"

            lo, hi = float(np.nanmin(ele)), float(np.nanmax(ele))
            span   = max(hi - lo, 1.0)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x, y=ele, mode="lines", name="Profile", showlegend=False,
                line=dict(color="rgba(140,140,140,0.85)", width=1.2),
                fill="tozeroy", fillcolor="rgba(140,140,140,0.10)",
                customdata=hhmm,
                hovertemplate=f"{x_hover} — %{{y:.0f}} m<extra></extra>",
            ))

            # Etichette dirette solo se poche: oltre 8 si sovrappongono
            label_direct = len(data) <= 8
            for i, entry in enumerate(data):
                df_seg = get_segment_df(entry)
                if df_seg is None or df_seg.empty:
                    continue
                idx   = df_seg.index
                seg_x = x.loc[idx]
                seg_y = ele.loc[idx]
                col   = colors[i % len(colors)]
                name  = entry.get("name", f"Segment {i+1}")

                fig.add_trace(go.Scatter(
                    x=seg_x, y=seg_y, mode="lines", name=name,
                    line=dict(color=col, width=2),
                    fill="tozeroy", fillcolor=_hex_rgba(col, 0.28),
                    customdata=hhmm.loc[idx],
                    hovertemplate=f"<b>{name}</b><br>{x_hover} — %{{y:.0f}} m<extra></extra>",
                ))
                if label_direct:
                    fig.add_annotation(
                        x=float((seg_x.iloc[0] + seg_x.iloc[-1]) / 2),
                        y=hi + 0.09 * span,
                        text=f"<b>{name}</b>", showarrow=False,
                        font=dict(size=11, color=_LABEL_INK),
                    )

            fig.update_layout(
                xaxis=dict(title=x_label, **_AXIS),
                yaxis=dict(title="Elevation (m)", tickformat="d",
                           range=[lo - 0.06 * span, hi + 0.18 * span], **_AXIS),
                hovermode="closest", height=420,
                margin=dict(t=60, b=50, l=60, r=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
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
            df_melted["HR Zone"] = df_melted["HR Zone"].str.replace("% ", "", regex=False)
            fig = px.bar(
                df_melted,
                x="HR Zone", y="Percentage",
                color=name_col, barmode="group",
                color_discrete_sequence=colors,
                title=title,
            )
            fig.update_traces(
                marker_line_width=0,
                hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.0f}%<extra></extra>",
            )
            fig.update_layout(
                barcornerradius=4, bargap=0.28, bargroupgap=0.08,
                xaxis=dict(title=None, **{**_AXIS, "showgrid": False}),
                yaxis=dict(title="% of time", ticksuffix="%", rangemode="tozero", **_AXIS),
                legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02,
                            xanchor="left", x=0),
                height=380, margin=dict(t=80, b=40, l=60, r=20),
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

                    climb_visible = pick_visible_columns(climb_df, climb_name_col, {
                        "❤️ Avg FC":       ["Avg FC"],
                        "📐 Grade":        ["Avg Grade (%)"],
                        "⛰️ VAM":          ["VAM (m/h)"],
                        "🏃 Speed":        ["Speed (km/h)"],
                        "⚡ EFS":          ["Avg EFS (km/h)"],
                        "🕒 Time in zone": zone_order,
                        "📊 % in zone":    pct_cols,
                    }, key_prefix="climb")

                    edited_climb_df = st.data_editor(
                        climb_visible,
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

                            adj_vam, tom_score = tom_low_score(vam, grade, avg_hr)

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

                    # --- Confronto salite con il curriculum ---
                    comparison.render_climbs(climb_df, st.session_state["climb_data"], df)

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

                lap_visible = pick_visible_columns(lap_df, lap_name_col, {
                    "❤️ Avg FC":       ["Avg FC"],
                    "⏱️ Pace":         ["Lap Pace (min/km)"],
                    "🏃 Speed":        ["Speed (km/h)"],
                    "⚡ EFS":          ["Avg EFS (km/h)"],
                    "🕒 Time in zone": zone_order,
                    "📊 % in zone":    pct_cols,
                }, key_prefix="lap")

                edited_lap_df = st.data_editor(
                    lap_visible,
                    num_rows="dynamic",
                    use_container_width=True,
                    disabled=True
                )
                st.session_state["lap_zone_data_edited"] = edited_lap_df

                show_zone_bar_chart(lap_df, lap_name_col, "🏃 Lap — % Time in HR Zones")

                # --- Confronto lap con il curriculum ---
                comparison.render_laps(lap_df)

            else:
                st.warning("⚠️ No lap data could be computed. Check your lap definitions.")

        elif st.session_state.get("do_lap_analysis"):
            st.info("👆 No lap data yet — define your laps in the section above.")