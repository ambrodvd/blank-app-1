"""7) Distribuzione di densità FC e heatmap tempo in zona."""
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from input_data import format_hmm, hhmm_to_minutes, hhmm_to_seconds


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "heatmap_df_minutes",
)


def render(ctx):
    analysis_ready = ctx.get("analysis_ready")
    bar_df = ctx.get("bar_df")
    df = ctx.get("df")
    uploaded_file = ctx.get("uploaded_file")
    z1 = ctx.get("z1")
    z2 = ctx.get("z2")
    z3 = ctx.get("z3")
    z4 = ctx.get("z4")
    z5 = ctx.get("z5")

    if analysis_ready and 'HR Zone' in df.columns and all(k in st.session_state for k in ['z1','z2','z3','z4','z5']):

        st.markdown("### 📊 Heart Rate Density Distribution by Zone")

        # --- Zone boundaries ---
        _zone_bands = [
            {"name": "Z1", "x0": 0,   "x1": z1, "color": "rgba(100, 200, 255, 0.15)"},
            {"name": "Z2", "x0": z1, "x1": z2, "color": "rgba(100, 220, 100, 0.15)"},
            {"name": "Z3", "x0": z2, "x1": z3, "color": "rgba(255, 230, 50,  0.15)"},
            {"name": "Z4", "x0": z3, "x1": z4, "color": "rgba(255, 150, 50,  0.15)"},
            {"name": "Z5", "x0": z4, "x1": z5, "color": "rgba(255, 80,  80,  0.15)"},
        ]

        DENSITY_CHART_HEIGHT = 160   # abbassa per stringere ancora
        def build_density_chart(hr_data, title, avg_hr, z1, z2, z3, z4, z5, zone_bands, x_min=None, x_max=None):
            kde = gaussian_kde(hr_data, bw_method=0.3)
            x_range = np.linspace(hr_data.min(), hr_data.max(), 500)
            y_kde = kde(x_range)
            y_kde = (y_kde / y_kde.sum()) * 100

            # Use provided range or fall back to data-derived range
            if x_min is None:
                x_min = np.percentile(hr_data, 1)
            if x_max is None:
                x_max = hr_data.max()

            fig = go.Figure()

            # Zone background bands
            for zone in zone_bands:
                fig.add_vrect(
                    x0=zone["x0"],
                    x1=zone["x1"],
                    fillcolor=zone["color"],
                    layer="below",
                    line_width=0,
                )
                band_center = (zone["x0"] + zone["x1"]) / 2
                if zone["name"] == "Z1":
                    label_x = z1 - 10
                else:
                    label_x = max(x_min, min(band_center, x_max))
                fig.add_annotation(
                    x=label_x,
                    y=1.02,
                    xref="x",
                    yref="paper",
                    text=zone["name"],
                    showarrow=False,
                    font=dict(size=15, color="black", weight="bold"),
                    xanchor="center"
                )

            # KDE smooth curve
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde,
                mode="lines",
                fill="tozeroy",
                line=dict(color="rgba(50, 120, 200, 1)", width=2),
                fillcolor="rgba(50, 120, 200, 0.3)",
                name="HR Density",
                hovertemplate="HR: %{x:.0f} bpm<br>Probability: %{y:.1f}%<extra></extra>"
            ))

            # Vertical zone boundary lines
            for bpm in [z1, z2, z3, z4, z5]:
                fig.add_vline(
                    x=bpm,
                    line_dash="dash",
                    line_color="gray",
                    line_width=1
                )

            # AVG RACE BPM line
            fig.add_vline(
                x=avg_hr,
                line_dash="solid",
                line_color="rgba(200, 50, 50, 1)",
                line_width=3
            )
            fig.add_annotation(
                x=avg_hr,
                y=0.97,
                xref="x",
                yref="paper",
                text=f"AVG RACE BPM<br><b>{avg_hr:.0f} bpm</b>",
                showarrow=False,
                font=dict(size=12, color="rgba(200, 50, 50, 1)"),
                bgcolor="white",
                bordercolor="rgba(200, 50, 50, 0.4)",
                borderwidth=1,
                xanchor="left",
                yanchor="top"
            )

            # Titolo dentro l'area del grafico e niente titolo asse x: sono i due
            # elementi che costano più spazio verticale, e con i grafici impilati
            # quello spazio è tutto scroll in più. L'annotazione va in alto a
            # sinistra, dove la densità è quasi sempre bassa.
            fig.add_annotation(
                x=0.01, y=0.97, xref="paper", yref="paper",
                text=f"<b>{title}</b>", showarrow=False,
                font=dict(size=12, color="black"),
                bgcolor="rgba(255,255,255,0.75)",
                xanchor="left", yanchor="top",
            )

            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Density %",
                showlegend=False,
                plot_bgcolor="white",
                height=DENSITY_CHART_HEIGHT,
                margin=dict(t=14, b=24, l=46, r=12),
                xaxis=dict(range=[x_min, x_max]),
                font=dict(size=11),
            )
            return fig

        # --- Compute full race avg ONCE ---
        _hr_data_total = df["heart_rate"].dropna()
        _avg_hr_total = _hr_data_total.mean()
        _x_min_total = np.percentile(_hr_data_total, 1)
        _x_max_total = _hr_data_total.max()

        # --- Full race chart ---
        if len(_hr_data_total) > 1:
            st.plotly_chart(
                build_density_chart(
                    _hr_data_total,
                    "Heart Rate Density Distribution - Full Race",
                    _avg_hr_total,
                    z1, z2, z3, z4, z5,
                    _zone_bands
                ),
                use_container_width=True
            )

        # --- Per-segment charts ---
        _segment_inputs = []
        for _i in range(1,  st.session_state.get('num_segments', 1) + 1):
            _start_key = f'segment{_i}_start'
            _end_key = f'segment{_i}_end'
            if all(k in st.session_state for k in [_start_key, _end_key]):
                _start_str = st.session_state[_start_key]
                _end_str = st.session_state[_end_key]
                _start_sec = hhmm_to_seconds(_start_str)
                _end_sec = hhmm_to_seconds(_end_str)
                if _start_sec is not None and _end_sec is not None and _end_sec > _start_sec:
                    _seg_name = f"{format_hmm(_start_str)} to {format_hmm(_end_str)}"
                    _segment_inputs.append((_start_sec, _end_sec, _seg_name))

        # Impilati, non affiancati: le distribuzioni si confrontano guardando
        # dove cade il picco sulla stessa scala x, e con due grafici a metà
        # larghezza separati da uno stacco l'occhio non ci riesce.
        for _start_sec, _end_sec, _seg_name in _segment_inputs:
            _df_seg = df[(df["elapsed_sec"] >= _start_sec) & (df["elapsed_sec"] <= _end_sec)]
            _hr_seg = _df_seg["heart_rate"].dropna()
            if len(_hr_seg) > 1:
                st.plotly_chart(
                    build_density_chart(
                        _hr_seg,
                        f"HR Density - {_seg_name}",
                        _avg_hr_total,
                        z1, z2, z3, z4, z5,
                        _zone_bands,
                        x_min=_x_min_total, x_max=_x_max_total
                    ),
                    use_container_width=True
                )
            else:
                st.warning(f"⚠️ Not enough HR data for segment {_seg_name}")

    if (uploaded_file is not None
            and bar_df is not None
            and not bar_df.empty):

        # Crea copia del DataFrame per sicurezza
        heatmap_df_minutes = bar_df.copy()

        # Applica la conversione a ogni cella di ogni colonna
        for col in heatmap_df_minutes.columns:
            heatmap_df_minutes[col] = heatmap_df_minutes[col].apply(hhmm_to_minutes)

        # Imposta l'indice con i nomi delle zone
        heatmap_df_minutes.index = [f"Zone {i+1}" for i in range(len(heatmap_df_minutes))]

        # Reset index e reshape per Plotly
        heatmap_df_minutes_reset = heatmap_df_minutes.reset_index().rename(columns={'index':'HR Zone'})
        heatmap_long = heatmap_df_minutes_reset.melt(
            id_vars="HR Zone",
            var_name="Segment",
            value_name="Minutes"
        )

        # Debug (opzionale) per verificare i dati
        # st.write("DEBUG heatmap_long:", heatmap_long.head())

        # Creazione heatmap con Plotly
        fig_heat = px.density_heatmap(
            heatmap_long,
            x="Segment",
            y="HR Zone",
            z="Minutes",
            text_auto=True,
            color_continuous_scale="YlOrRd",
            hover_data={"Segment": True, "HR Zone": True, "Minutes": True},
            title="🌡️ Time-in-Zone (minutes) Heatmap"
        )

        # Inverti l'asse y per convenzione zone
        fig_heat.update_layout(
            yaxis=dict(autorange='reversed'),
            coloraxis_colorbar=dict(title="Time (minutes)")
        )

        # Mostra heatmap su Streamlit
        st.plotly_chart(fig_heat, use_container_width=True)

    else:
        st.warning(" ⚠️ Please submit all data required for the analysis.")

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})
