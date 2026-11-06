"""DU COACHING RACE Analyzer — entry point (app a pagina singola).

Le sezioni vengono eseguite in ordine, dall'alto verso il basso.
Ogni file espone render(ctx): `ctx` è il dizionario condiviso in cui ogni
sezione legge le variabili prodotte prima (df, z1..z5, analysis_ready, ...)
e scrive quelle che servono dopo (elenco in _EXPORTS in cima a ogni render).

    1) input_data.py            upload FIT, dati atleta, zone cardiache
    2a) segments_laps.py        segmenti temporali, scelta Lap/Climb, lap analysis
    2b) climbs.py               climb analysis
    3) mail_info.py             invio dati via email + riepilogo
    4) det_graphs.py            DET index, grafico live, speed decadence
    5) efficiency.py            Efficiency Factor
    6) segment_analysis.py      tempo in zona per segmento
    7) density_heatmap.py       densità FC e heatmap
    8) lap_climb_report.py      tabelle e grafici lap / climb
    9) pdf_builder.py           report PDF
   10) curriculum.py            storico gare dell'atleta (file JSON)
   11) comparison.py            confronto con le gare del curriculum (dentro ogni sezione)
"""
import base64
import os

import streamlit as st

import input_data
import segments_laps
import climbs
import mail_info
import det_graphs
import efficiency
import segment_analysis
import density_heatmap
import lap_climb_report
import pdf_builder
import curriculum
import comparison
from pdf_builder import LOGO_DU_PATH, SCRITTA_PATH

st.set_page_config(page_title="DU COACHING RACE Analyzer", layout="wide")

# --- Intestazione: i due loghi affiancati e centrati, poi il titolo ---
# Le immagini sono inlined in base64 dentro un unico blocco HTML flex:
# st.columns + st.image non permette di centrare davvero (ogni immagine si
# centra nella propria colonna) e lascia margini verticali non controllabili.
LOGO_DU_HEIGHT_PX = 180    # stessa altezza per entrambi -> allineamento ottico
SCRITTA_HEIGHT_PX = 164    # la scritta è più "leggera", va un filo più bassa

def _img_tag(path: str, height_px: int) -> str:
    if not os.path.exists(path):
        return ""
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (f"<img src='data:{mime};base64,{b64}' "
            f"style='height:{height_px}px; width:auto; display:block;'/>")

st.markdown(
    "<div style='display:flex; align-items:center; justify-content:center; "
    "gap:28px; margin:0 0 6px 0;'>"
    + _img_tag(LOGO_DU_PATH, LOGO_DU_HEIGHT_PX)
    + _img_tag(SCRITTA_PATH, SCRITTA_HEIGHT_PX)
    + "</div>"
    "<h1 style='text-align:center; margin:0 0 0.4em 0; padding-top:0;'>"
    "📊 DU COACHING RACE Analyzer 📊</h1>",
    unsafe_allow_html=True
)

st.info("This analyzer is brought to you by coach Davide Ambrosini")
st.markdown(
    """
    <div style="background-color:#7dd1d0; padding:10px; border-left:5px solid #a01fb4; border-radius:5px; color:#2c18db">
    This app is free but if you want to support my work you can 
    <a href="https://buymeacoffee.com/ultranerd" target="_blank" style="color:#a01fb4; font-weight:bold;">buy me a coffee</a>
    </div>
    """,
    unsafe_allow_html=True
)

ctx = {}
st.session_state["cmp_pdf"] = {}   # confronti da aggiungere al PDF: li riempiono le sezioni

curriculum.render_loader(ctx)     # 10a — curriculum (facoltativo) in cima
comparison.render_top()           # 11  — gare del curriculum + scelta di default
input_data.render(ctx)            # 1
segments_laps.render_setup(ctx)   # 2a — segmenti + checkbox Lap/Climb
climbs.render(ctx)                # 2b
segments_laps.render_laps(ctx)    # 2a — lap analysis
mail_info.render(ctx)             # 3
det_graphs.render(ctx)            # 4
efficiency.render(ctx)            # 5
segment_analysis.render(ctx)      # 6
density_heatmap.render(ctx)       # 7
lap_climb_report.render(ctx)      # 8

### COACH COMMENT SECTION ###

st.subheader("Coach Comment Section")

# Initialize the comment in session state
if "comment" not in st.session_state:
    st.session_state.comment = ""

# Text input for the user
comment_input = st.text_area("Insert your comment here:", value=st.session_state.comment)

# Save button
if st.button("Save Comment"):
    st.session_state.comment = comment_input
    st.success("Comment saved!")

# Display the stored comment
if st.session_state.comment:
    st.subheader("Coach Comment:")
    st.write(st.session_state.comment)

pdf_builder.render(ctx)           # 9
curriculum.render(ctx)            # 10b — salvataggio curriculum
