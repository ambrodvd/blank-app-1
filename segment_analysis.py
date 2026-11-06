"""6) Segment analysis: tempo in zona per segmento."""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from climbs import _hex_rgba
from input_data import (
    HR_ZONE_LABELS,
    CHART_PALETTE,
    format_hmm,
    hhmm_to_minutes,
    hhmm_to_seconds,
    seconds_to_hhmm,
)


# Stile assi comune ai grafici: griglia e bordi recessivi
_AXIS = dict(showgrid=True, gridcolor="rgba(128,128,128,0.18)",
             showline=True, linecolor="rgba(128,128,128,0.55)",
             mirror=True, ticks="outside", ticklen=4, zeroline=False)


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "bar_df", "combined_df", "z1", "z2", "z3", "z4", "z5",
)


def render(ctx):
    analysis_ready = ctx.get("analysis_ready")
    df = ctx.get("df")

    if analysis_ready:
    # ------------------------------------------
    # SEGMENT ANALYSIS #
    # ------------------------------------------

        st.markdown("## SEGMENT ANALYSIS")

        # Time in zone analysis for segments

        # Check conditions

        if all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):
            z1, z2, z3, z4, z5 = st.session_state['z1'], st.session_state['z2'], st.session_state['z3'], st.session_state['z4'], st.session_state['z5']

            # "HR Zone" è già calcolata in input_data.py.
            # time_diff_sec va ricalcolato: il lap detector automatico può aver
            # sostituito df con una versione senza le righe prive di GPS.
            df["time_diff_sec"] = df["elapsed_sec"].diff().clip(lower=0).fillna(0)

            zone_order = HR_ZONE_LABELS

            # Total (overall) time-in-zone
            total_summary = df.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)

            # Time segment warning

            _n = st.session_state.get('num_segments', 1)
            segment_keys = [f'segment{i}_{se}' for i in range(1, _n+1) for se in ['start','end']]

            if not all(k in st.session_state for k in segment_keys):
                missing_seg = [k for k in segment_keys if k not in st.session_state]
                st.warning(f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis.")

            # Prepare segment inputs with formatted names
            segment_inputs = []    
            for i in range(1, st.session_state.get('num_segments', 1) + 1):
                start_key = f'segment{i}_start'
                end_key = f'segment{i}_end'
                if all(k in st.session_state for k in [start_key, end_key]):
                    seg_name = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                    segment_inputs.append(
                        (st.session_state[start_key], st.session_state[end_key], seg_name)
                    )

            segment_data = {}


            for start_str, end_str, seg_name in segment_inputs:
                start_sec = hhmm_to_seconds(start_str)
                end_sec = hhmm_to_seconds(end_str)
                if start_sec is None or end_sec is None or start_sec >= end_sec:
                    segment_data[seg_name] = pd.Series(0, index=zone_order)
                    continue

                df_segment = df[(df["elapsed_sec"] >= start_sec) & (df["elapsed_sec"] <= end_sec)].copy()
                df_segment["time_diff_sec"] = df_segment["elapsed_sec"].diff().fillna(0)
                seg_summary = df_segment.groupby("HR Zone")["time_diff_sec"].sum().reindex(zone_order).fillna(0)
                segment_data[seg_name] = seg_summary

            # Combine all segments + total into one DataFrame
            combined_df = pd.DataFrame(segment_data)
            combined_df["Total"] = total_summary

            # Format only numeric columns as H:MM
            num_cols = combined_df.select_dtypes(include=["number"]).columns
            for col in num_cols:
                combined_df[col] = combined_df[col].apply(
                    lambda x: f"{int(x//3600)}:{int((x%3600)//60):02d}" if pd.notna(x) else ""
                )

            st.markdown("### ⏱️ Time-in-Zone Analysis")
            st.dataframe(combined_df)

        else:
            st.warning("⚠️ Please submit the Heart Rate Zones to enable Time-in-Zone analysis.")    


    # --- Elevation Profile with Time Segments (Live Chart) ---
    if analysis_ready and df is not None and not df.empty:

        # Stesso stile dei grafici Lap / Climb: quota lisciata, asse y stretto,
        # asse x numerico in ore, segmenti con linea piena + riempimento chiaro.
        x_h    = df["elapsed_sec"] / 3600.0
        x_hhmm = df["elapsed_sec"].apply(seconds_to_hhmm)
        elevation_smooth = df["elevation_m"].rolling(window=25, center=True, min_periods=1).mean()
        lo, hi = float(np.nanmin(elevation_smooth)), float(np.nanmax(elevation_smooth))
        span   = max(hi - lo, 1.0)

        fig_seg = go.Figure()

        # Profilo completo in grigio, sotto
        fig_seg.add_trace(go.Scatter(
            x=x_h, y=elevation_smooth, mode="lines", name="Profile", showlegend=False,
            line=dict(color="rgba(140,140,140,0.85)", width=1.2),
            fill="tozeroy", fillcolor="rgba(140,140,140,0.10)",
            customdata=x_hhmm,
            hovertemplate="%{customdata} — %{y:.0f} m<extra></extra>",
        ))

        n_seg = st.session_state.get('num_segments', 1)
        for i in range(1, n_seg + 1):
            start_key = f'segment{i}_start'
            end_key   = f'segment{i}_end'
            if all(k in st.session_state for k in [start_key, end_key]):
                seg_start = hhmm_to_seconds(st.session_state[start_key])
                seg_end   = hhmm_to_seconds(st.session_state[end_key])
                if seg_start is not None and seg_end is not None and seg_end > seg_start:
                    mask = (df["elapsed_sec"] >= seg_start) & (df["elapsed_sec"] <= seg_end)
                    if not mask.any():
                        continue
                    seg_label = f"{format_hmm(st.session_state[start_key])} to {format_hmm(st.session_state[end_key])}"
                    color = CHART_PALETTE[(i - 1) % len(CHART_PALETTE)]
                    fig_seg.add_trace(go.Scatter(
                        x=x_h[mask], y=elevation_smooth[mask], mode="lines", name=seg_label,
                        line=dict(color=color, width=2),
                        fill="tozeroy", fillcolor=_hex_rgba(color, 0.28),
                        customdata=x_hhmm[mask],
                        hovertemplate=f"<b>{seg_label}</b><br>%{{customdata}} — %{{y:.0f}} m<extra></extra>",
                    ))
                    # Etichetta sopra il profilo, al centro del segmento
                    if n_seg <= 8:
                        fig_seg.add_annotation(
                            x=float((x_h[mask].iloc[0] + x_h[mask].iloc[-1]) / 2),
                            y=hi + 0.09 * span, text=f"<b>{seg_label}</b>", showarrow=False,
                            font=dict(size=11, color="rgba(110,110,110,1)"),
                        )

        fig_seg.update_layout(
            title="Elevation Profile with Time Segments Highlighted",
            xaxis=dict(title="Elapsed time (h)", **_AXIS),
            yaxis=dict(title="Elevation (m)", tickformat="d",
                       range=[lo - 0.06 * span, hi + 0.18 * span], **_AXIS),
            hovermode="closest", height=420,
            margin=dict(t=90, b=50, l=60, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )

        st.plotly_chart(fig_seg, use_container_width=True)

        # GROUPED BAR CHART FOR TIME IN ZONE
    if analysis_ready and 'HR Zone' in df.columns:
        missing_seg = [k for k in segment_keys if k not in st.session_state]
        if missing_seg:
            st.warning(
                f"⚠️ Please submit the Time Segments in the form above to enable Time-in-Zone analysis"
            )

        bar_df = combined_df.copy()
        bar_df.index = [f"Zone {i+1}" for i in range(len(bar_df))]
        bar_df_reset = bar_df.reset_index().rename(columns={'index':'HR Zone'})
        bar_long = bar_df_reset.melt(id_vars="HR Zone", var_name="Segment", value_name="Time [h:mm]")

        # H:MM -> ore decimali
        bar_long["Hours"] = bar_long["Time [h:mm]"].apply(hhmm_to_minutes) / 60

        bar_long["HR Zone"] = bar_long["HR Zone"].str.replace("Zone ", "Z", regex=False)

        # Colore fisso per segmento (lo stesso del profilo), Total in grigio neutro
        _seg_cols  = [c for c in bar_df.columns if c != "Total"]
        _color_map = {c: CHART_PALETTE[j % len(CHART_PALETTE)] for j, c in enumerate(_seg_cols)}
        _color_map["Total"] = "#9a9a9a"

        fig_bar = px.bar(
            bar_long,
            x="HR Zone",
            y="Hours",
            color="Segment",
            barmode="group",
            color_discrete_map=_color_map,
            custom_data=["Segment", "Time [h:mm]"],
            title="⏱️ Time-in-Zone per Segment",
        )
        fig_bar.update_traces(
            marker_line_width=0,
            hovertemplate="<b>%{customdata[0]}</b><br>%{x}: %{customdata[1]} h:mm<extra></extra>",
        )
        fig_bar.update_layout(
            barcornerradius=4, bargap=0.28, bargroupgap=0.08,
            xaxis=dict(title=None, **{**_AXIS, "showgrid": False}),
            yaxis=dict(title="Hours", rangemode="tozero", **_AXIS),
            legend=dict(title=None, orientation="h", yanchor="bottom", y=1.02,
                        xanchor="left", x=0),
            height=380, margin=dict(t=80, b=40, l=60, r=20),
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})