"""2a) Segmenti temporali, scelta Lap/Climb e Lap analysis."""
import math

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from input_data import seconds_to_hhmm


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS_SETUP = (
    "do_climb", "do_lap",
)


def render_setup(ctx):
    with st.form("num_segments_form"):
        st.subheader("⏱️ Time segments for Time in Zone Analysis")
        st.caption("Step 1: Choose how many time segments you want to analyze")

        num_segments = st.number_input(
            "How many time segments do you want to analyze?",
            min_value=1, max_value=10, value=st.session_state.get('num_segments', 1), step=1
        )

        num_segments_submitted = st.form_submit_button("Save Number of Segments")

    if num_segments_submitted:
        st.session_state['num_segments'] = num_segments
        st.success(f"✅ {num_segments} segment(s) selected!")

    # --- Step 2: Input Segment Lengths (only shown after number is saved) ---
    if st.session_state.get('num_segments'):
        with st.form("time_segment_form"):
            st.caption("Step 2: Define the start and end time (H:MM) for each segment")

            # --- Compute total duration safely ---
            total_duration_sec = None
            try:
                if 'fit_df' in st.session_state:
                    _df = st.session_state['fit_df']
                    if 'elapsed_sec' in _df.columns and not _df['elapsed_sec'].isna().all():
                        total_duration_sec = float(_df['elapsed_sec'].iloc[-1])
            except Exception:
                total_duration_sec = None

            def default_segment_time(i, n, total_sec, boundary="start"):
                """Return HH:MM string for segment i (1-indexed) boundary."""
                try:
                    if total_sec is None or total_sec <= 0 or n <= 0:
                        raise ValueError
                    segment_sec = total_sec / n
                    if boundary == "start":
                        secs = segment_sec * (i - 1)
                    else:
                        secs = segment_sec * i
                    secs = int(round(secs))
                    h = secs // 3600
                    m = (secs % 3600) // 60
                    return f"{h}:{m:02d}"
                except Exception:
                    return "0:00" if boundary == "start" else "1:00"

            n = st.session_state['num_segments']
            segment_inputs = {}

            for i in range(1, n + 1):
                col1, col2 = st.columns(2)

                default_start = default_segment_time(i, n, total_duration_sec, "start")
                default_end   = default_segment_time(i, n, total_duration_sec, "end")

                with col1:
                    segment_inputs[f'segment{i}_start'] = st.text_input(
                        f"Segment {i} Start",
                        value=st.session_state.get(f'segment{i}_start', default_start)
                    )
                with col2:
                    segment_inputs[f'segment{i}_end'] = st.text_input(
                        f"Segment {i} End",
                        value=st.session_state.get(f'segment{i}_end', default_end)
                    )

            segments_submitted = st.form_submit_button("Save Time Segments")

        if segments_submitted:
            # Clear old segments first (in case user reduced the count)
            for i in range(1, 11):
                st.session_state.pop(f'segment{i}_start', None)
                st.session_state.pop(f'segment{i}_end', None)

            for key, val in segment_inputs.items():
                st.session_state[key] = val
            st.success("✅ Time segments saved successfully!")

    st.markdown("## 📋 Lap / Climb Analysis")
    st.markdown("### Which analysis do you want to perform?")

    # -----------------------------------------------------------
    # 1️⃣ SELECT ANALYSIS TYPES (independent toggles)
    # -----------------------------------------------------------

    col_lap, col_climb = st.columns(2)

    with col_lap:
        do_lap = st.checkbox(
            "🏃 Lap Analysis",
            value=st.session_state.get("do_lap_analysis", False),
            key="do_lap_checkbox"
        )

    with col_climb:
        do_climb = st.checkbox(
            "⛰️ Climb Analysis",
            value=st.session_state.get("do_climb_analysis", False),
            key="do_climb_checkbox"
        )

    st.session_state["do_lap_analysis"] = do_lap
    st.session_state["do_climb_analysis"] = do_climb

    # Keep a combined flag for downstream code that checks "do_lap_analysis"
    # (lap OR climb active means some analysis is running)
    st.session_state["any_analysis_active"] = do_lap or do_climb

    # Reset climb insert method if climb is disabled
    if not do_climb:
        st.session_state["climb_data_insert"] = None

    # Reset lap insert method if lap is disabled
    if not do_lap:
        st.session_state["lap_data_insert"] = None

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS_SETUP if k in _loc})


# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS_LAPS = (
    "df",
)


def render_laps(ctx):
    do_lap = ctx.get("do_lap")
    uploaded_file = ctx.get("uploaded_file")

    if do_lap:
        st.markdown("---")
        st.markdown("### 🏃 Lap Analysis")

        lap_data_insert = st.radio(
            "How do you want to insert lap data?",
            ("Manually", "Automatic Lap detector", "Distance slicer"),
            index=0,
            key="lap_data_insert_selector"
        )

        if lap_data_insert == "Manually":
            st.session_state["lap_data_insert"] = "manual"
        elif lap_data_insert == "Automatic Lap detector":
            st.session_state["lap_data_insert"] = "automatic"
        else:
            st.session_state["lap_data_insert"] = "fix_distance"

        # ---------------------------
        # AUTOMATIC LOOP DETECTOR
        # ---------------------------
        loops = []

        if st.session_state.get("lap_data_insert") == "automatic":
            if uploaded_file is None or "fit_df" not in st.session_state:
                st.warning("👆 Please upload a .fit file first.")
            else:
                df = st.session_state["fit_df"]

                col1, col2, col3 = st.columns(3)
                with col1:
                    base_radius = st.number_input("Base radius (meters)", value=20, min_value=1)
                with col2:
                    percent_error = st.slider("GPS antenna error (%)", 0.0, 5.0, 1.0) / 100
                with col3:
                    min_samples_between_crossings = st.slider("Min samples between crossings", 1, 50, 5)

                required_cols = ["lat", "lon", "timestamp", "elevation_m", "distance_km", "elapsed_sec"]
                if not all(col in df.columns for col in required_cols):
                    st.error(f"Missing required columns: {', '.join([c for c in required_cols if c not in df.columns])}")
                else:
                    df = df.dropna(subset=["lat", "lon", "timestamp"]).reset_index(drop=True)
                    df["elevation_m"] = df["elevation_m"].astype(float)
                    df["distance_km"] = df["distance_km"].astype(float)

                    from haversine import haversine

                    start_lat, start_lon = df.loc[0, ["lat", "lon"]]
                    df["dist_from_start_m"] = df.apply(
                        lambda r: haversine((start_lat, start_lon), (r["lat"], r["lon"])) * 1000,
                        axis=1
                    )
                    threshold = base_radius + percent_error * df["dist_from_start_m"].max()

                    crossings = []
                    inside = df.loc[0, "dist_from_start_m"] <= threshold
                    last_crossing_idx = 0 if inside else -1
                    if inside:
                        crossings.append((0, df.loc[0, "timestamp"]))

                    for idx in range(1, len(df)):
                        d = df.loc[idx, "dist_from_start_m"]
                        currently_inside = d <= threshold
                        if (not inside) and currently_inside:
                            if (idx - last_crossing_idx) > min_samples_between_crossings:
                                crossings.append((idx, df.loc[idx, "timestamp"]))
                                last_crossing_idx = idx
                        inside = currently_inside

                    if len(crossings) > 1:
                        for i in range(1, len(crossings)):
                            prev_idx, _ = crossings[i - 1]
                            idx, _ = crossings[i]
                            df_lap = df.iloc[prev_idx:idx + 1].copy()
                            start_sec = df_lap["elapsed_sec"].iloc[0]
                            end_sec = df_lap["elapsed_sec"].iloc[-1]
                            duration_sec = end_sec - start_sec
                            loops.append({
                                "name": f"Lap {i}",
                                "start_time": seconds_to_hhmm(start_sec),
                                "end_time": seconds_to_hhmm(end_sec),
                                "duration": seconds_to_hhmm(duration_sec),
                                "start_idx": prev_idx,
                                "end_idx": idx,
                                "distance": round(df_lap["distance_km"].max() - df_lap["distance_km"].min(), 1),
                                "elevation": int(df_lap["elevation_m"].diff().clip(lower=0).sum()),
                            })

                    st.session_state["lap_data"] = loops
                    st.session_state["lap_form_submitted"] = True

                    st.subheader("Detected Loops")
                    if loops:
                        loops_df = pd.DataFrame(loops)
                        loops_df["distance"] = loops_df["distance"].astype(float).map("{:.1f}".format)
                        st.dataframe(loops_df[["name", "start_time", "end_time", "duration", "distance", "elevation"]])
                    else:
                        st.warning("No loops detected.")

        # ---------------------------
        # DISTANCE SLICER
        # ---------------------------
        elif st.session_state.get("lap_data_insert") == "fix_distance":
            if uploaded_file is None or "fit_df" not in st.session_state:
                st.info("👆 Please upload a .fit file first to use the Distance Slicer.")
            else:
                df = st.session_state["fit_df"]
                max_distance = df["distance_km"].max()
                slice_distance = st.number_input(
                    "How many km would you like each lap to be?",
                    min_value=1,
                    max_value=int(math.ceil(max_distance)),
                    value=10
                )

                if st.button("Generate Laps", key="generate_distance_laps"):
                    laps = []
                    start_dist = 0.0
                    lap_num = 1
                    total_distance = round(max_distance, 3)

                    while start_dist < total_distance:
                        end_dist = min(start_dist + slice_distance, total_distance)
                        lap_df = df[
                            (df["distance_km"].round(3) >= round(start_dist, 3)) &
                            (df["distance_km"].round(3) <= round(end_dist, 3))
                        ].copy()
                        if lap_df.empty:
                            break

                        start_sec = lap_df["elapsed_sec"].iloc[0]
                        end_sec = lap_df["elapsed_sec"].iloc[-1]
                        duration_sec = end_sec - start_sec

                        laps.append({
                            "name": f"Lap {lap_num}",
                            "start_time": seconds_to_hhmm(start_sec),
                            "end_time": seconds_to_hhmm(end_sec),
                            "duration": seconds_to_hhmm(duration_sec),
                            "start_idx": lap_df.index[0],
                            "end_idx": lap_df.index[-1],
                            "distance": round(end_dist - start_dist, 3)
                        })
                        start_dist += slice_distance
                        lap_num += 1

                    st.session_state["lap_data"] = laps
                    st.session_state["lap_form_submitted"] = True

                    st.subheader("Distance Slicer Laps")
                    if laps:
                        st.dataframe(pd.DataFrame(laps)[["name", "start_time", "end_time", "duration", "distance"]])
                    else:
                        st.warning("No laps generated. Check the distance slice or your FIT file.")

    # ---------------------------
    # MANUAL LAP INPUT (distance-based)
    # ---------------------------

        elif st.session_state.get("lap_data_insert") == "manual":
                if "fit_df" not in st.session_state:
                    st.warning("👆 Please upload a .fit file first.")
                else:
                    df = st.session_state["fit_df"]
                    max_distance = round(df["distance_km"].max(), 2)

                    def nearest_idx_by_distance(df, target_km):
                        return int((df["distance_km"] - target_km).abs().idxmin())

                    # --- Elevation profile (x = distance) ---
                    st.subheader("Lap Elevation Profile")
                    fig_manual = go.Figure()
                    fig_manual.add_trace(go.Scatter(
                        x=df["distance_km"],
                        y=df["elevation_m"].astype(int),
                        mode="lines",
                        line=dict(color="gray"),
                        hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y} m<extra></extra>"
                    ))
                    fig_manual.update_layout(
                        xaxis_title="Distance (km)",
                        yaxis_title="Elevation (m)"
                    )
                    st.plotly_chart(fig_manual, use_container_width=True)
                    st.caption(f"Total race distance: **{max_distance} km**")

                    # --- CSV import ---
                    st.markdown("#### 📥 Import Laps from CSV")
                    imported_lap_csv = st.file_uploader(
                        "Import previously exported lap CSV:",
                        type=["csv"],
                        key="import_lap_csv"
                    )
                    if imported_lap_csv is not None:
                        try:
                            imported_df = pd.read_csv(imported_lap_csv)
                            required_import_cols = {"name", "start_km", "end_km"}
                            if not required_import_cols.issubset(set(imported_df.columns)):
                                st.error(f"⚠️ CSV must contain columns: name, start_km, end_km")
                            else:
                                imported_df = imported_df[["name", "start_km", "end_km"]]
                                imported_df["name"] = imported_df["name"].fillna("").astype(str)
                                st.session_state["manual_lap_table"] = imported_df.reset_index(drop=True)
                                st.success(f"✅ Imported {len(imported_df)} lap(s) from CSV!")
                        except Exception as e:
                            st.error(f"⚠️ Error reading CSV: {e}")

                    # --- Editable table ---
                    table_key = "manual_lap_table"
                    if table_key not in st.session_state:
                        st.session_state[table_key] = pd.DataFrame({
                            "name":     pd.Series([], dtype="str"),
                            "start_km": pd.Series([], dtype="float"),
                            "end_km":   pd.Series([], dtype="float"),
                        })

                    st.subheader("Add/Edit Laps")
                    st.info(f"Enter the start and end distance in km for each lap (max: {max_distance} km)")

                    with st.form("manual_lap_form"):
                                    edited = st.data_editor(
                                        st.session_state[table_key],
                                        num_rows="dynamic",
                                        key="manual_lap_editor",
                                        column_config={
                                            "name": st.column_config.TextColumn("Lap Name"),
                                            "start_km": st.column_config.NumberColumn(
                                                "Start (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                                            ),
                                            "end_km": st.column_config.NumberColumn(
                                                "End (km)", min_value=0.0, max_value=float(max_distance), step=0.1, format="%.2f"
                                            ),
                                        }
                                    )
                                    submit = st.form_submit_button("Save manual Laps")
                    if submit:
                        edited = edited.reset_index(drop=True)

                        # --- Auto-name any unnamed rows first ---
                        edited["name"] = [
                            row["name"] if pd.notna(row.get("name")) and str(row.get("name", "")).strip() != ""
                            else f"Lap {i+1}"
                            for i, row in edited.iterrows()
                        ]

                        # --- Check race name is available for export ---
                        race_name_ok = bool(st.session_state.get("race_name", "").strip())
                        if not race_name_ok:
                            st.warning("⚠️ Race name is not set — please submit the Race Info form above to enable CSV export.")

                        # --- Validation: start_km and end_km required ---
                        incomplete_rows = []
                        for i, row in edited.iterrows():
                            if pd.isna(row.get("start_km")):
                                incomplete_rows.append(f"Row {i+1} ({row['name']}): missing Start (km)")
                            if pd.isna(row.get("end_km")):
                                incomplete_rows.append(f"Row {i+1} ({row['name']}): missing End (km)")
                            elif not pd.isna(row.get("start_km")) and float(row["end_km"]) <= float(row["start_km"]):
                                incomplete_rows.append(f"Row {i+1} ({row['name']}): End km must be greater than Start km")

                        if incomplete_rows:
                            for msg in incomplete_rows:
                                st.error(f"⚠️ {msg}")
                            st.stop()

                        if not edited.empty:
                            st.session_state[table_key] = edited

                        lap_data = []
                        errors = []
                        for i, row in edited.iterrows():
                            start_km = float(row["start_km"])
                            end_km   = float(row["end_km"])

                            if end_km > max_distance:
                                errors.append(f"Row {i+1} ({row['name']}): end km ({end_km}) exceeds race distance ({max_distance} km).")
                                continue

                            start_idx    = nearest_idx_by_distance(df, start_km)
                            end_idx      = nearest_idx_by_distance(df, end_km)
                            start_sec    = df.loc[start_idx, "elapsed_sec"]
                            end_sec      = df.loc[end_idx,   "elapsed_sec"]
                            duration_sec = end_sec - start_sec
                            lap_slice    = df.loc[start_idx:end_idx]
                            elevation    = int(lap_slice["elevation_m"].diff().clip(lower=0).sum())

                            lap_data.append({
                                "name":       row["name"],
                                "start_km":   round(start_km, 2),
                                "end_km":     round(end_km, 2),
                                "distance":   round(end_km - start_km, 2),
                                "start_time": seconds_to_hhmm(start_sec),
                                "end_time":   seconds_to_hhmm(end_sec),
                                "duration":   seconds_to_hhmm(duration_sec),
                                "elevation":  elevation,
                                "start_idx":  start_idx,
                                "end_idx":    end_idx,
                            })

                        for err in errors:
                            st.error(err)

                        if lap_data:
                            st.session_state["lap_data"] = lap_data
                            st.session_state["lap_form_submitted"] = True
                            st.session_state["lap_export_ready"] = True
                            st.session_state["lap_export_df"] = edited[["name", "start_km", "end_km"]].copy()
                            st.success(f"✅ {len(lap_data)} lap(s) saved!")
                        else:
                            st.session_state["lap_export_ready"] = False
                            st.error("⚠️ No valid laps to save. Please check your distance values.")

                    # --- CSV export (outside form) ---
                    if st.session_state.get("lap_export_ready"):
                        race_name_ok = bool(st.session_state.get("race_name", "").strip())
                        if race_name_ok:
                            csv_bytes = st.session_state["lap_export_df"].to_csv(index=False).encode("utf-8")
                            safe_race_name = st.session_state["race_name"].strip().replace(" ", "_")
                            st.download_button(
                                label="📥 Download Laps as CSV",
                                data=csv_bytes,
                                file_name=f"{safe_race_name}_lap_analyzer.csv",
                                mime="text/csv",
                                key="download_lap_csv"
                            )
                        else:
                            st.info("ℹ️ Submit the Race Info form above to enable CSV export.")
                    else:
                        st.error("⚠️ No valid laps to save. Please check your distance values.")

                    # --- Preview: highlight laps on elevation profile ---
                    if st.session_state.get("lap_form_submitted") and st.session_state.get("lap_data"):
                        fig_final = go.Figure()
                        fig_final.add_trace(go.Scatter(
                            x=df["distance_km"],
                            y=df["elevation_m"].astype(int),
                            mode="lines",
                            line=dict(color="gray"),
                            hovertemplate="Distance: %{x:.2f} km<br>Elevation: %{y} m<extra></extra>"
                        ))

                        for entry in st.session_state["lap_data"]:
                            s = entry["start_idx"]
                            e = entry["end_idx"]
                            df_segment = df.loc[s:e]
                            if df_segment.empty:
                                continue
                            fig_final.add_trace(go.Scatter(
                                x=df_segment["distance_km"],
                                y=df_segment["elevation_m"].astype(int),
                                mode="lines",
                                line=dict(color="green"),
                                fill="tozeroy",
                                opacity=0.4,
                                name=entry["name"],
                                hovertemplate=f"{entry['name']}<br>Distance: %{{x:.2f}} km<br>Elevation: %{{y}} m<extra></extra>"
                            ))

                        fig_final.update_layout(
                            title="Manual Laps Highlighted",
                            xaxis_title="Distance (km)",
                            yaxis_title="Elevation (m)",
                            hovermode="x unified",
                            showlegend=True
                        )
                        st.plotly_chart(fig_final, use_container_width=True)

                        display_cols = ["name", "start_km", "end_km", "distance", "start_time", "end_time", "duration", "elevation"]
                        st.dataframe(
                            pd.DataFrame(st.session_state["lap_data"])[[c for c in display_cols]].reset_index(drop=True)
                        )

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS_LAPS if k in _loc})
