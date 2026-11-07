"""3) Invio dati via email e riepilogo atleta/gara."""
import os
import smtplib
from email.message import EmailMessage

import streamlit as st
import pandas as pd

import comparison
from det_graphs import compute_efs_series


# ======================================================================
# HELPERS
# ======================================================================
def build_archive_payload(fit_bytes, fit_filename, anonymous=False):
    """Ritorna [(filename, bytes, mimetype), ...] con il FIT originale e il
    CSV delle zone FC rigenerato da session_state. Il CSV si ricostruisce
    invece di riusare l'upload perché in modalità Manual Input il file non
    esiste, e comunque le zone possono essere state modificate a mano."""
    ext = ".fit.gz" if fit_filename.lower().endswith(".gz") else os.path.splitext(fit_filename)[1]

    if anonymous:
        # Niente nome, gara o data: né nei filename né dentro il CSV.
        # La data di gara da sola, incrociata con i risultati pubblici,
        # basterebbe a risalire all'atleta.
        out = [(f"race_data{ext}", fit_bytes, "application/octet-stream")]
        zone_row = {}
    else:
        athlete = (st.session_state.get("athlete_name") or "athlete").strip().replace(" ", "_")
        race    = (st.session_state.get("race_name") or "race").strip().replace(" ", "_")
        stamp   = st.session_state.get("race_date")
        stamp   = stamp.strftime("%Y%m%d") if stamp else "nodate"
        prefix  = f"{athlete}_{race}_{stamp}"
        out = [(f"{prefix}{ext}", fit_bytes, "application/octet-stream")]
        zone_row = {"athlete_name": st.session_state.get("athlete_name", "")}

    if all(k in st.session_state for k in ("z1", "z2", "z3", "z4", "z5")):
        zone_row.update({
            "z1": st.session_state["z1"], "z2": st.session_state["z2"],
            "z3": st.session_state["z3"], "z4": st.session_state["z4"],
            "z5": st.session_state["z5"],
        })
        zones_csv = pd.DataFrame([zone_row]).to_csv(index=False).encode("utf-8")
        csv_name = "hr_zones.csv" if anonymous else f"{prefix}_HR_Zones.csv"
        out.append((csv_name, zones_csv, "text/csv"))

    return out


def missing_archive_requirements():
    """Lista dei dati mancanti per l'invio. Vuota = si può inviare.
    Il file .fit non è qui dentro: quel controllo lo fa già il blocco
    chiamante, che senza upload non disegna nemmeno il bottone."""
    missing = []
    if not str(st.session_state.get("athlete_name", "")).strip():
        missing.append("Athlete's Name")
    if not str(st.session_state.get("race_name", "")).strip():
        missing.append("Race name")
    if st.session_state.get("race_date") is None:
        missing.append("Race date")
    if not st.session_state.get("hr_zones_confirmed"):
        missing.append("HR Zones (press 'Submit HR Zones')")
    return missing


def missing_anonymous_requirements():
    """Per l'invio anonimo servono solo le zone FC: nome atleta, gara e
    data non finiscono da nessuna parte, quindi non ha senso pretenderli."""
    if not st.session_state.get("hr_zones_confirmed"):
        return ["HR Zones (press 'Submit HR Zones')"]
    return []


def send_payload_email(payload, subject, body):
    """SMTP con STARTTLS. L'eccezione risale al chiamante: l'errore va
    mostrato all'utente con st.error, non ingoiato qui dentro."""
    cfg = st.secrets["email"]
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(body)

    for fname, data, mime in payload:
        maintype, _, subtype = mime.partition("/")
        msg.add_attachment(data, maintype=maintype, subtype=subtype or "octet-stream",
                           filename=fname)

    with smtplib.SMTP(cfg["smtp_host"], int(cfg["smtp_port"]), timeout=60) as s:
        s.starttls()
        s.login(cfg["user"], cfg["password"])
        s.send_message(msg)
    return len(payload)


# ======================================================================
# RENDER
# ======================================================================
# Variabili prodotte qui e lette dalle sezioni successive
_EXPORTS = (
    "final_time_str", "first_half_avg", "formatted_date", "overall_avg", "percent_diff",
    "second_half_avg",
)


def render(ctx):
    analysis_ready = ctx.get("analysis_ready")
    df = ctx.get("df")
    kilometers = ctx.get("kilometers")
    total_elevation_gain = ctx.get("total_elevation_gain")
    uploaded_file = ctx.get("uploaded_file")
    z1 = ctx.get("z1")
    z2 = ctx.get("z2")
    z3 = ctx.get("z3")
    z4 = ctx.get("z4")
    z5 = ctx.get("z5")

    st.markdown(
        """
    <hr style="border:1px solid #336699">
    <h4 style='text-align:center; color:white;'>📊 Race Analysis</h4>
    <hr style="border:1px solid #336699">
    """,
        unsafe_allow_html=True
    )

    # ----- analysis start -----#
    # --- Archiviazione file grezzi via email ---------------------------------
    # Nessun invio automatico: parte solo al click. Il bottone resta disabilitato
    # finché non ci sono sia il file sia i dati di gara: un FIT che arriva senza
    # nome atleta è inutilizzabile in archivio.
    if uploaded_file is not None:
        _missing = missing_archive_requirements()

        st.markdown(
            """
        <div style="background-color:#fff3cd; padding:10px; border-left:5px solid #e0a800; border-radius:5px; color:#7a5c00; margin-bottom:10px;">
        ⚠️ If you wanna help the development of this app, please consider sending the developer the race and cardiac data you're using on this analysis clicking the button below ⚠️
        </div>
        """,
            unsafe_allow_html=True
        )

        st.info(
            "You can choose between sending the full data or an anonymized version. The full data will "
            "be stored in the developer's archive for future analysis and development, while the "
            "'anonymized' version will be used for testing and improving the app without any personal information."
        )

        _send_col, _spacer = st.columns([2, 4])
        _subject = (f"[RACE ANALYZER] App data for {st.session_state.get('athlete_name', '?')} - "
                    f"{st.session_state.get('race_name', '?')}")
        _body = (f"Athlete: {st.session_state.get('athlete_name', '')}\n"
                 f"Race: {st.session_state.get('race_name', '')}\n"
                 f"Distance: {st.session_state.get('kilometers', '?')} km\n"
                 f"D+: {st.session_state.get('total_elevation_gain', '?')} m\n")

        _clicked = _send_col.button(
            "📧 Send raw files by email",
            key="send_files_email",
            disabled=bool(_missing),
            help=("Missing: " + ", ".join(_missing)) if _missing else
                 "Sends the raw .fit file and the HR zones CSV",
        )

        if _missing:
            st.caption("⚠️ Before sending, please complete: " + ", ".join(_missing))

        if _clicked:
            # Ricontrollo al click: fra il rendering del bottone e la pressione
            # può esserci stato un rerun che ha svuotato qualcosa.
            _missing = missing_archive_requirements()
            if _missing:
                st.error("❌ Missing data: " + ", ".join(_missing))
            else:
                try:
                    _payload = build_archive_payload(uploaded_file.getvalue(), uploaded_file.name)
                    _n = send_payload_email(_payload, _subject, _body)
                    st.success(f"✅ Email sent with {_n} attachment(s).")
                except Exception as e:
                    st.error(f"❌ Email failed: {e}")


        # --- Invio anonimo ---------------------------------------------------
        # Stessi allegati, ma senza alcun riferimento all'atleta: filename
        # generici e corpo del messaggio vuoto. Il bottone è separato invece
        # di essere una checkbox perché la scelta va fatta consapevolmente.
        _anon_missing = missing_anonymous_requirements()
        _anon_col, _anon_spacer = st.columns([2, 4])

        _anon_clicked = _anon_col.button(
            "🕶️ Send anonymized data",
            key="send_files_email_anon",
            disabled=bool(_anon_missing),
            help=("Missing: " + ", ".join(_anon_missing)) if _anon_missing else
                 "Sends the same files with no name, race or date attached",
        )

        if _anon_clicked:
            _anon_missing = missing_anonymous_requirements()
            if _anon_missing:
                st.error("❌ Missing data: " + ", ".join(_anon_missing))
            else:
                try:
                    _payload = build_archive_payload(
                        uploaded_file.getvalue(), uploaded_file.name, anonymous=True)
                    _n = send_payload_email(_payload, "[RACE ANALYZER] Anonymous data", "")
                    st.success(f"✅ Anonymous data sent ({_n} attachment(s)).")
                except Exception as e:
                    st.error(f"❌ Email failed: {e}")

    if not analysis_ready:
        if uploaded_file is None:
            st.info("👆 Upload a FIT file to run the analysis")
        else:
            st.warning(
                "🔒 **Analysis locked** — press **Submit HR Zones** above to confirm "
                "the athlete's heart rate zones."
            )
    else:
        # --- Athlete & race info display ---
        if 'athlete_name' not in st.session_state:
            st.warning("⚠️ Please submit the Athlete and Race info in the form above")
        else:
            st.markdown("---")
            st.markdown(f"**Athlete:** {st.session_state['athlete_name']}")
            st.markdown(f"**Race:** {st.session_state['race_name']}")
            formatted_date = st.session_state['race_date'].strftime("%d/%m/%Y")
            st.markdown(f"**Date:** {formatted_date}")
            st.markdown(f"📏 **Distance Run:** {kilometers} km")
            st.markdown(f"🏔️ **Elevation gain:** {total_elevation_gain} m")

            total_seconds = df["elapsed_sec"].iloc[-1]
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            final_time_str = f"{hours}:{minutes:02d}:{seconds:02d}"
            st.markdown(f"**Final Time:** {final_time_str}")
            st.markdown("---")
            st.write(f"""
        **HR Zones:**  
        - 🩵 Zone 1 (Aerobic Low): ≤ {z1} bpm  
        - 💚 Zone 2 (Aerobic High): {z1+1} - {z2} bpm  
        - 💛 Zone 3 (Aerobic Endurance): {z2+1} - {z3} bpm  
        - 🧡 Zone 4 (Sub Threshold): {z3+1} - {z4} bpm  
        - ❤️ Zone 5 (Super Threshold): {z4+1} - {z5} bpm
        """)
            st.markdown("---")

        # --- Smooth HR ---
        # Già calcolata nel parser, ma va rifatta: il lap detector automatico
        # può aver sostituito df con una versione senza le righe prive di GPS.
        df["hr_smooth"] = df["heart_rate"].rolling(window=3, min_periods=1).mean()

        # --- HR averages ---
        mid_index = len(df) // 2
        first_half_avg = df["heart_rate"][:mid_index].mean()
        second_half_avg = df["heart_rate"][mid_index:].mean()
        overall_avg = df["heart_rate"].mean()
        percent_diff = ((second_half_avg - first_half_avg) / first_half_avg) * 100

        st.markdown(f"❤️ Overall Average HR: **{overall_avg:.0f} bpm**")
        st.markdown(f"🟢 First Half Average: **{first_half_avg:.0f} bpm**")
        st.markdown(f"🔵 Second Half Average: **{second_half_avg:.0f} bpm**")
        if percent_diff >= -10:
            st.success(f"📊 % Difference: **{percent_diff:.1f}%**")
        else:
            st.error(f"📊 % Difference: **{percent_diff:.1f}%**")

        # --- FIT giusto per la gara riaperta? ---
        comparison.check_reopened_fit()

        # --- Confronto con il curriculum ---
        comparison.render_total(
            df, overall_avg, percent_diff,
            avg_efs_fn=lambda: compute_efs_series(df)[1].get("avg_efs_kmh"),
        )

    # --- export verso le sezioni successive ---
    _loc = locals()
    ctx.update({k: _loc[k] for k in _EXPORTS if k in _loc})