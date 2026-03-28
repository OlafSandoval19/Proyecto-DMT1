import streamlit as st
import pandas as pd
from datetime import datetime, date, time
from pathlib import Path
import json

from kidia.state import init_state
from kidia.auth import logout
from kidia.ui import render_kidia_header

# =========================
# 0) CONFIG DE PÁGINA
# =========================
st.set_page_config(page_title="KIDIA | Ingestas manuales", layout="wide")
init_state()

if not st.session_state.get("authenticated", False):
    st.switch_page("app.py")

render_kidia_header()

# =========================
# CONSTANTES
# =========================
MAX_EVENTOS = 5
BASE_UPLOADS = Path("data/uploads")
PATIENTS_CSV = Path("data/patients.csv")
MANUAL_DATA_DIR = Path("data/manual_prediction")
MANUAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# CSS / ESTILOS
# =========================
st.markdown("""
<style>
    [data-testid="stSidebarNav"] ul li:first-child {
        display: none;
    }

    [data-testid="stSidebarNav"]::before {
        content: "Menú principal";
        display: block;
        font-size: 1.35rem;
        font-weight: 700;
        color: #1f2a44;
        margin: 0.5rem 0 1rem 0.3rem;
        padding-top: 0.5rem;
    }

    .sidebar-bottom-space {
        height: 55vh;
    }

    .block-container {
        max-width: 100% !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-top: 1.2rem !important;
    }

    .data-title {
        margin-top: 0rem !important;
        margin-bottom: 0.3rem !important;
        text-align: center;
        font-size: 2.1rem;
        font-weight: 800;
        color: #1f2a44;
    }

    .data-subtitle {
        text-align: center;
        font-size: 1.05rem;
        font-weight: 600;
        color: #374151;
        margin-top: 0;
        margin-bottom: 1.5rem;
        letter-spacing: 0.2px;
        line-height: 1.4;
    }

    .patient-section {
        width: 100% !important;
        padding: 0.25rem 0 0.4rem 0 !important;
        margin-bottom: 0.8rem !important;
    }

    button[data-baseweb="tab"] p {
        font-size: 1.08rem !important;
        font-weight: 800 !important;
    }

    button[data-baseweb="tab"] {
        padding: 14px 18px !important;
        height: auto !important;
        flex: 1 1 0% !important;
        justify-content: center !important;
    }

    div[data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        width: 100% !important;
        justify-content: stretch !important;
        gap: 10px !important;
    }

    div[data-testid="stTabs"] [role="tab"] {
        flex: 1 1 0% !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #ef4444 !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR LIMPIO
# =========================
with st.sidebar:
    st.markdown('<div class="sidebar-bottom-space"></div>', unsafe_allow_html=True)
    if st.button("Cerrar sesión", use_container_width=True):
        logout()
        st.rerun()

# =========================
# HELPERS
# =========================
def patient_label(p: dict) -> str:
    return f'{p.get("patient_name", "Sin nombre")} ({p.get("patient_id", "Sin ID")})'

def get_registered_patients(base_uploads: Path, patients_csv: Path):
    rows = []

    if patients_csv.exists():
        try:
            df = pd.read_csv(patients_csv)
            if not df.empty:
                for _, r in df.iterrows():
                    pid = str(r.get("ID", r.get("patient_id", ""))).strip()
                    pname = str(r.get("Nombre", r.get("patient_name", ""))).strip()

                    if pid:
                        folder_path = base_uploads / f"patient_{pid}"
                        rows.append({
                            "patient_id": pid,
                            "patient_name": pname if pname else pid,
                            "folder_path": str(folder_path)
                        })
        except Exception:
            pass

    # respaldo por si el CSV no existe o viene vacío
    if not rows and base_uploads.exists():
        for p in sorted(base_uploads.glob("patient_*")):
            pid = p.name.replace("patient_", "")
            rows.append({
                "patient_id": pid,
                "patient_name": pid,
                "folder_path": str(p)
            })

    return rows

def get_manual_patient_dir(patient_id: str) -> Path:
    pdir = MANUAL_DATA_DIR / str(patient_id)
    pdir.mkdir(parents=True, exist_ok=True)
    return pdir

def get_events_csv_path(patient_id: str) -> Path:
    return get_manual_patient_dir(patient_id) / "eventos_manuales.csv"

def get_config_json_path(patient_id: str) -> Path:
    return get_manual_patient_dir(patient_id) / "config_pronostico.json"

def load_saved_events(patient_id: str):
    csv_path = get_events_csv_path(patient_id)
    if not csv_path.exists():
        return []

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []

        expected_cols = [
            "tipo", "fecha", "hora", "datetime",
            "cho_valor", "cho_unidad", "cho_mg",
            "bolus_u", "nota"
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = ""

        return df.to_dict(orient="records")
    except Exception:
        return []

def save_events_csv(patient_id: str, events: list):
    csv_path = get_events_csv_path(patient_id)

    ordered_cols = [
        "tipo", "fecha", "hora", "datetime",
        "cho_valor", "cho_unidad", "cho_mg",
        "bolus_u", "nota"
    ]

    if not events:
        pd.DataFrame(columns=ordered_cols).to_csv(csv_path, index=False, encoding="utf-8-sig")
        return csv_path

    df = pd.DataFrame(events).copy()
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[ordered_cols]
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path

def save_prediction_config_json(patient_id: str, config: dict):
    json_path = get_config_json_path(patient_id)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return json_path

def _safe_time_from_any(value) -> time:
    if isinstance(value, time):
        return value

    if value is None:
        return time(0, 0)

    txt = str(value).strip()
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(txt, fmt).time()
        except Exception:
            pass

    dt = pd.to_datetime(txt, errors="coerce")
    if pd.notna(dt):
        return dt.time()

    return time(0, 0)

def normalize_entries_to_reference_date(entries: list, pred_date):
    """
    Reasigna TODOS los eventos a la fecha de referencia actual,
    conservando la hora capturada.
    """
    normalized = []

    if not entries:
        return normalized

    for ev in entries:
        ev2 = dict(ev)

        hora_obj = _safe_time_from_any(ev2.get("hora", "00:00"))
        event_dt = datetime.combine(pred_date, hora_obj)

        cho_unidad = str(ev2.get("cho_unidad", "g")).strip().lower()
        cho_valor = float(ev2.get("cho_valor", 0.0) or 0.0)

        if "cho_mg" in ev2 and ev2.get("cho_mg") not in ["", None]:
            try:
                cho_mg = float(ev2.get("cho_mg", 0.0) or 0.0)
            except Exception:
                cho_mg = cho_valor * 1000.0 if cho_unidad == "g" else cho_valor
        else:
            cho_mg = cho_valor * 1000.0 if cho_unidad == "g" else cho_valor

        ev2["tipo"] = ev2.get("tipo", "Evento prandial")
        ev2["fecha"] = pred_date.strftime("%Y-%m-%d")
        ev2["hora"] = hora_obj.strftime("%H:%M")
        ev2["datetime"] = event_dt.strftime("%Y-%m-%d %H:%M:%S")
        ev2["cho_valor"] = float(cho_valor)
        ev2["cho_unidad"] = "g" if cho_unidad not in ["g", "mg"] else cho_unidad
        ev2["cho_mg"] = float(cho_mg) if cho_mg > 0 else 0.0
        ev2["bolus_u"] = float(ev2.get("bolus_u", 0.0) or 0.0)
        ev2["nota"] = str(ev2.get("nota", "") or "").strip()

        normalized.append(ev2)

    normalized.sort(key=lambda x: x.get("datetime", ""))
    return normalized

def build_editable_events_df(entries: list, pred_date, start_time_value, active_rows=1):
    rows = []
    entries = normalize_entries_to_reference_date(entries, pred_date)

    active_rows = max(1, min(int(active_rows), MAX_EVENTOS))

    for ev in entries[:active_rows]:
        rows.append({
            "hora": str(ev.get("hora", "08:00"))[:5],
            "cho_valor": float(ev.get("cho_valor", 0.0) or 0.0),
            "cho_unidad": str(ev.get("cho_unidad", "g") or "g"),
            "bolus_u": float(ev.get("bolus_u", 0.0) or 0.0),
            "nota": str(ev.get("nota", "") or ""),
        })

    while len(rows) < active_rows:
        rows.append({
            "hora": "08:00",
            "cho_valor": 0.0,
            "cho_unidad": "g",
            "bolus_u": 0.0,
            "nota": "",
        })

    return pd.DataFrame(rows)


def validate_and_convert_events_df(df_edit: pd.DataFrame, pred_date, start_time_value):
    errors = []
    cleaned_entries = []

    if df_edit is None or df_edit.empty:
        return cleaned_entries, errors

    horas_vistas = set()

    for idx, row in df_edit.iterrows():
        hora_raw = str(row.get("hora", "") or "").strip()
        cho_valor = row.get("cho_valor", 0.0)
        cho_unidad = str(row.get("cho_unidad", "g") or "g").strip().lower()
        bolus_u = row.get("bolus_u", 0.0)
        nota = str(row.get("nota", "") or "").strip()

        try:
            cho_valor = float(cho_valor or 0.0)
        except Exception:
            cho_valor = 0.0

        try:
            bolus_u = float(bolus_u or 0.0)
        except Exception:
            bolus_u = 0.0

        fila_vacia = (hora_raw == "") and (cho_valor <= 0) and (bolus_u <= 0) and (nota == "")
        if fila_vacia:
            continue

        if hora_raw == "":
            errors.append(f"Fila {idx + 1}: debes indicar una hora.")
            continue

        hora_obj = _safe_time_from_any(hora_raw)
        hora_fmt = hora_obj.strftime("%H:%M")

        if hora_fmt in horas_vistas:
            errors.append(f"Fila {idx + 1}: la hora {hora_fmt} está duplicada. No se permiten eventos prandiales solapados.")
            continue

        horas_vistas.add(hora_fmt)

        if cho_unidad not in ["g", "mg"]:
            cho_unidad = "g"

        if cho_valor < 0:
            errors.append(f"Fila {idx + 1}: los carbohidratos no pueden ser negativos.")
            continue

        if bolus_u < 0:
            errors.append(f"Fila {idx + 1}: la insulina no puede ser negativa.")
            continue

        if cho_valor == 0 and bolus_u == 0:
            errors.append(f"Fila {idx + 1}: agrega carbohidratos o bolo, de lo contrario deja la fila vacía.")
            continue

        event_dt = datetime.combine(pred_date, hora_obj)
        cho_mg = cho_valor * 1000.0 if cho_unidad == "g" else cho_valor

        cleaned_entries.append({
            "tipo": "Evento prandial",
            "fecha": pred_date.strftime("%Y-%m-%d"),
            "hora": hora_fmt,
            "datetime": event_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "cho_valor": float(cho_valor),
            "cho_unidad": cho_unidad,
            "cho_mg": float(cho_mg) if cho_valor > 0 else 0.0,
            "bolus_u": float(bolus_u),
            "nota": nota,
        })

    cleaned_entries.sort(key=lambda x: x["datetime"])

    if len(cleaned_entries) > MAX_EVENTOS:
        errors.append(f"Solo se permiten hasta {MAX_EVENTOS} eventos por día.")

    return cleaned_entries[:MAX_EVENTOS], errors


def save_all_manual_data(patient_id: str, patient_name: str, patient_folder: str,
                         pred_date, start_time_value, glucose_now, entries: list):
    # CORRECCIÓN CLAVE:
    # los eventos se re-fechan SIEMPRE a la fecha de referencia actual
    entries_normalized = normalize_entries_to_reference_date(entries, pred_date)

    # actualizamos el state también
    st.session_state.manual_entries_by_patient[str(patient_id)] = entries_normalized

    csv_path = save_events_csv(patient_id, entries_normalized)

    df_entries = pd.DataFrame(entries_normalized) if entries_normalized else pd.DataFrame()
    config_payload = {
        "patient_id": patient_id,
        "patient_name": patient_name,
        "patient_folder": patient_folder,
        "reference_date": str(pred_date),
        "start_time": start_time_value.strftime("%H:%M"),
        "horizon_minutes": 1440,
        "current_glucose_mgdl": float(glucose_now),
        "events_csv_path": str(csv_path),
        "events": df_entries.to_dict(orient="records") if not df_entries.empty else [],
    }

    json_path = save_prediction_config_json(patient_id, config_payload)
    st.session_state.manual_prediction_config = config_payload
    st.session_state.forecast_locked_patient_id = str(patient_id)
    st.session_state.forecast_locked_patient_name = str(patient_name)
    return csv_path, json_path, config_payload

def sync_manual_data_after_edit(patient_id: str, patient_name: str, patient_folder: str,
                                pred_date, start_time_value, glucose_now, entries: list):
    csv_path, json_path, config_payload = save_all_manual_data(
        patient_id=patient_id,
        patient_name=patient_name,
        patient_folder=patient_folder,
        pred_date=pred_date,
        start_time_value=start_time_value,
        glucose_now=glucose_now,
        entries=entries
    )
    st.session_state.manual_prediction_config = config_payload
    return csv_path, json_path

def render_confidence_message(n_eventos: int, max_eventos: int = 6):
    if n_eventos <= 0:
        st.info("ℹ️ Aún no hay eventos capturados. Agrega eventos para construir un escenario de pronóstico.")
    elif n_eventos == max_eventos:
        st.success(
            "✅ Se alcanzó el máximo recomendado de 5 eventos. "
            "Este escenario es el más consistente con la dinámica usada en el entrenamiento."
        )
    elif n_eventos == 5:
        st.info(
            "ℹ️ Hay 5 eventos capturados. La confiabilidad sigue siendo buena, "
            "aunque ligeramente menor respecto a un escenario de 5 eventos."
        )
    elif n_eventos == 4:
        st.warning(
            "⚠️ Hay 4 eventos capturados. La confiabilidad del pronóstico puede disminuir "
            "porque el escenario empieza a alejarse del patrón de entrenamiento."
        )
    elif n_eventos == 3:
        st.warning(
            "⚠️ Hay 3 eventos capturados. La confiabilidad del pronóstico disminuye de forma moderada, "
            "ya que el modelo fue entrenado con una dinámica de hasta 5 eventos."
        )
    elif n_eventos == 2:
        st.error(
            "🚨 Solo hay 2 eventos capturados. La confiabilidad del pronóstico puede ser baja "
            "por alejarse notablemente del escenario de entrenamiento."
        )
    elif n_eventos == 1:
        st.error(
            "🚨 Solo hay 1 evento capturado. La confiabilidad del pronóstico es baja "
            "y la simulación puede no representar adecuadamente la dinámica diaria."
        )
    else:
        st.error(
            f"🚨 Hay {n_eventos} eventos capturados. Este escenario está fuera del rango esperado "
            "y la confiabilidad puede verse comprometida."
        )

def generate_time_options(step_minutes=15):
    options = []
    for h in range(24):
        for m in range(0, 60, step_minutes):
            options.append(f"{h:02d}:{m:02d}")
    return options

TIME_OPTIONS = generate_time_options(15)




# =========================
# STATE LOCAL
# =========================
if "manual_entries_by_patient" not in st.session_state:
    st.session_state.manual_entries_by_patient = {}

if "manual_prediction_config" not in st.session_state:
    st.session_state.manual_prediction_config = {}

if "manual_loaded_patients" not in st.session_state:
    st.session_state.manual_loaded_patients = set()

# =========================
# TÍTULO
# =========================
st.markdown("""
<h2 class="data-title">
    💉 Ingestas manuales 🍽️
</h2>
<p class="data-subtitle">
    Selecciona un paciente, registra ingestas e insulina y explora los eventos capturados para el pronóstico.
</p>
""", unsafe_allow_html=True)

# =========================
# 1) PACIENTES REGISTRADOS
# =========================
patients = get_registered_patients(BASE_UPLOADS, PATIENTS_CSV)

if not patients:
    st.error("No se encontraron pacientes registrados en 'data/uploads'.")
    st.stop()

patients_df = pd.DataFrame(patients)

# =========================
# 2) SELECCIÓN DE PACIENTE
# =========================
st.markdown('<div class="patient-section">', unsafe_allow_html=True)
st.markdown("### 👤 Selección de paciente")

f1, f2 = st.columns(2)

with f1:
    use_patient_id_filter = st.checkbox("Buscar por ID", key="manual_use_patient_id_filter")
    patient_id_filter = st.text_input(
        "Filtrar por ID",
        placeholder="Ejemplo: CHILD_001",
        key="manual_patient_id_filter",
        disabled=not use_patient_id_filter
    )

with f2:
    use_patient_name_filter = st.checkbox("Buscar por Nombre", key="manual_use_patient_name_filter")
    patient_name_filter = st.text_input(
        "Filtrar por Nombre",
        placeholder="Ejemplo: Niño 1",
        key="manual_patient_name_filter",
        disabled=not use_patient_name_filter
    )

patients_filtered = patients_df.copy()

if use_patient_id_filter and patient_id_filter.strip():
    term_id = patient_id_filter.strip().lower()
    patients_filtered = patients_filtered[
        patients_filtered["patient_id"].astype(str).str.lower().str.contains(term_id, na=False)
    ]

if use_patient_name_filter and patient_name_filter.strip():
    term_name = patient_name_filter.strip().lower()
    patients_filtered = patients_filtered[
        patients_filtered["patient_name"].astype(str).str.lower().str.contains(term_name, na=False)
    ]

patients_filtered = patients_filtered.reset_index(drop=True)

st.caption(f"Pacientes encontrados: {len(patients_filtered)}")

if patients_filtered.empty:
    st.warning("No se encontraron pacientes con los filtros seleccionados.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

filtered_options = patients_filtered.to_dict(orient="records")

selected_patient = st.selectbox(
    "Selecciona el paciente registrado",
    options=filtered_options,
    format_func=patient_label,
    key="manual_selected_registered_patient"
)

st.markdown('</div>', unsafe_allow_html=True)

patient_id = selected_patient["patient_id"]
patient_name = selected_patient["patient_name"]
patient_folder = selected_patient["folder_path"]
patient_key = str(patient_id)

# =========================
# CARGA AUTOMÁTICA DE EVENTOS GUARDADOS
# =========================
if patient_key not in st.session_state.manual_entries_by_patient:
    st.session_state.manual_entries_by_patient[patient_key] = []

if patient_key not in st.session_state.manual_loaded_patients:
    saved_events = load_saved_events(patient_key)
    st.session_state.manual_entries_by_patient[patient_key] = saved_events
    st.session_state.manual_loaded_patients.add(patient_key)

# =========================
# BOX DE PACIENTE ACTIVO
# =========================
events_csv_path = get_events_csv_path(patient_key)

st.markdown(
    f"""
    <div style="
        padding: 14px 16px;
        border-radius: 14px;
        background: rgba(46, 204, 113, 0.12);
        border: 1px solid rgba(46, 204, 113, 0.35);
        font-size: 18px;
        line-height: 1.35;
        margin-bottom: 18px;
    ">
        <div style="font-size: 20px; font-weight: 800;">
            🟢 Paciente activo: {patient_name} <span style="font-weight:600; opacity:0.9;">(ID {patient_id})</span>
        </div>
        <div style="margin-top: 6px; font-size: 15px; opacity: 0.85;">
            Paciente listo para registrar eventos manuales y configurar el pronóstico.
        </div>
        <div style="margin-top: 6px; font-size: 14px; opacity: 0.8;">
            Carpeta de guardado: {events_csv_path.parent}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# 3) TABS
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "👤 Paciente",
    "⚙️ Configuración",
    "📋 Eventos",
    "💾 Guardar eventos"
])

# =========================
# TAB 1: PACIENTE
# =========================
with tab1:
    st.subheader("Paciente registrado")
    st.info("El paciente se obtiene del registro en el apartado GESTIÓN DE PACIENTES.")

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**Nombre:** {patient_name}")
    with c2:
        st.write(f"**ID:** {patient_id}")

# =========================
# TAB 2: CONFIGURACIÓN
# =========================
with tab2:
    st.subheader("Configuración general de predicción")

    saved_config_path = get_config_json_path(patient_key)

    default_date = date.today()
    default_start_time = time(0, 0)
    default_glucose = 120.0

    if saved_config_path.exists():
        try:
            with open(saved_config_path, "r", encoding="utf-8") as f:
                cfg_saved = json.load(f)

            if "reference_date" in cfg_saved:
                default_date = datetime.strptime(cfg_saved["reference_date"], "%Y-%m-%d").date()

            if "start_time" in cfg_saved:
                default_start_time = datetime.strptime(cfg_saved["start_time"], "%H:%M").time()

            if "current_glucose_mgdl" in cfg_saved:
                default_glucose = float(cfg_saved["current_glucose_mgdl"])
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pred_date = st.date_input(
            "Fecha de referencia",
            value=default_date,
            key=f"manual_pred_date_{patient_key}"
        )

    with c2:
        start_time_value = st.time_input(
            "Hora inicial de simulación",
            value=default_start_time,
            key=f"manual_start_time_{patient_key}"
        )

    with c3:
        horizon_minutes = 1440
        horizon_hours = horizon_minutes / 60
        horizon_days = horizon_minutes / 1440

        st.metric(
            "Horizonte",
            f"{horizon_minutes} min | {horizon_hours:.1f} h | {horizon_days:.1f} día"
        )

    with c4:
        glucose_now = st.number_input(
            "Glucosa inicial (mg/dL)",
            min_value=0.0,
            max_value=600.0,
            value=float(default_glucose),
            step=1.0,
            key=f"manual_current_glucose_{patient_key}"
        )

    st.info("Si cambias la fecha de referencia, al guardar se ajustarán automáticamente las fechas de todos los eventos, conservando sus horas.")

# =========================
# TAB 3: EVENTOS
# =========================
with tab3:
    st.subheader("Gestión de eventos")
    st.markdown("### ➕ Tabla editable de eventos del día")
    st.caption(f"Puedes capturar hasta {MAX_EVENTOS} eventos prandiales. No se permiten horas duplicadas.")

    entries = st.session_state.manual_entries_by_patient.get(patient_key, [])
    entries = normalize_entries_to_reference_date(entries, pred_date)
    st.session_state.manual_entries_by_patient[patient_key] = entries

    count_key = f"manual_event_count_{patient_key}"
    if count_key not in st.session_state:
        st.session_state[count_key] = max(1, len(entries)) if entries else 1

    b_add, b_remove = st.columns(2)

    with b_add:
        if st.button("➕ Agregar evento", use_container_width=True, key=f"btn_add_row_{patient_key}"):
            if st.session_state[count_key] < MAX_EVENTOS:
                st.session_state[count_key] += 1
                st.rerun()

    with b_remove:
        if st.button("➖ Quitar evento", use_container_width=True, key=f"btn_remove_row_{patient_key}"):
            if st.session_state[count_key] > 1:
                st.session_state[count_key] -= 1

                current_entries = st.session_state.manual_entries_by_patient.get(patient_key, [])
                st.session_state.manual_entries_by_patient[patient_key] = current_entries[:st.session_state[count_key]]

                st.rerun()

    df_editor_base = build_editable_events_df(
        entries,
        pred_date,
        start_time_value,
        active_rows=st.session_state[count_key]
    )

    edited_df = st.data_editor(
        df_editor_base,
        use_container_width=True,
        num_rows="fixed",
        hide_index=True,
        key=f"manual_events_editor_{patient_key}",
        column_config={
            "hora": st.column_config.SelectboxColumn(
                "Hora",
                help="Selecciona la hora del evento",
                options=TIME_OPTIONS,
                required=True,
                width="small",
            ),
            "cho_valor": st.column_config.NumberColumn(
                "Carbohidratos",
                min_value=0.0,
                max_value=1000.0,
                step=1.0,
                format="%.2f",
                width="small",
            ),
            "cho_unidad": st.column_config.SelectboxColumn(
                "Unidad CHO",
                options=["g", "mg"],
                width="small",
            ),
            "bolus_u": st.column_config.NumberColumn(
                "Dosis de insulina (U)",
                min_value=0.0,
                max_value=100.0,
                step=0.1,
                format="%.2f",
                width="small",
            ),
            "nota": st.column_config.TextColumn(
                "Nota opcional",
                width="medium",
            ),
        }
    )

    if st.button("💾 Aplicar cambios de la tabla", type="primary", use_container_width=True, key=f"apply_table_events_{patient_key}"):
        cleaned_entries, errors = validate_and_convert_events_df(edited_df, pred_date, start_time_value)

        if errors:
            for err in errors:
                st.error(err)
        else:
            st.session_state.manual_entries_by_patient[patient_key] = cleaned_entries
            st.session_state[count_key] = max(1, len(cleaned_entries)) if cleaned_entries else 1

            sync_manual_data_after_edit(
                patient_id=patient_key,
                patient_name=patient_name,
                patient_folder=patient_folder,
                pred_date=pred_date,
                start_time_value=start_time_value,
                glucose_now=glucose_now,
                entries=cleaned_entries
            )

            st.success("Eventos actualizados y guardados correctamente.")
            st.rerun()

    st.divider()
    st.markdown("### 📋 Eventos capturados")

    entries = st.session_state.manual_entries_by_patient[patient_key]
    entries = normalize_entries_to_reference_date(entries, pred_date)
    st.session_state.manual_entries_by_patient[patient_key] = entries

    render_confidence_message(len(entries), MAX_EVENTOS)

    if not entries:
        st.info("Aún no has agregado eventos.")
    else:
        df_entries = pd.DataFrame(entries).copy()

        if "datetime" in df_entries.columns:
            df_entries["datetime_dt"] = pd.to_datetime(df_entries["datetime"], errors="coerce")
            df_entries = df_entries.sort_values("datetime_dt").reset_index(drop=True)

            start_dt = datetime.combine(pred_date, start_time_value)
            df_entries["entra_horizonte"] = df_entries["datetime_dt"] >= start_dt
            df_entries["datetime"] = df_entries["datetime_dt"].dt.strftime("%Y-%m-%d %H:%M")

        df_show = df_entries.copy()
        show_cols = [c for c in ["hora", "cho_valor", "cho_unidad", "bolus_u", "nota", "datetime", "entra_horizonte"] if c in df_show.columns]
        st.dataframe(df_show[show_cols], use_container_width=True)

        if df_entries["hora"].duplicated().any():
            st.error("Se detectaron horas duplicadas. Corrige la tabla para evitar eventos solapados.")

        if any(pd.to_datetime(df_entries["datetime_dt"], errors="coerce") < pd.Timestamp(datetime.combine(pred_date, start_time_value))):
            st.warning("Hay eventos antes de la hora inicial del pronóstico; esos eventos podrían no entrar al horizonte simulado.")

        if st.button("🧹 Limpiar todos los eventos", use_container_width=True, key=f"delete_all_{patient_key}"):
            st.session_state.manual_entries_by_patient[patient_key] = []
            st.session_state[count_key] = 1

            sync_manual_data_after_edit(
                patient_id=patient_key,
                patient_name=patient_name,
                patient_folder=patient_folder,
                pred_date=pred_date,
                start_time_value=start_time_value,
                glucose_now=glucose_now,
                entries=[]
            )

            st.success("Todos los eventos fueron eliminados y cambios guardados.")
            st.rerun()

    st.divider()
    st.markdown("### 📊 Resumen")

    if entries:
        df_entries = pd.DataFrame(entries)
        total_cho_g = df_entries["cho_mg"].fillna(0).sum() / 1000.0 if "cho_mg" in df_entries.columns else 0.0
        total_bolus_u = df_entries["bolus_u"].fillna(0).sum() if "bolus_u" in df_entries.columns else 0.0

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Eventos", int(len(df_entries)))
        r2.metric("Total CHO (g)", f"{total_cho_g:,.2f}")
        r3.metric("Total bolo (U)", f"{total_bolus_u:,.2f}")
        r4.metric("Glucosa inicial", f"{glucose_now:,.1f} mg/dL")
        r5.metric("Hora inicial", start_time_value.strftime("%H:%M"))
    else:
        st.info("Agrega al menos un evento para ver el resumen.")

# =========================
# TAB 4: GUARDAR / PRONÓSTICO
# =========================
with tab4:
    st.subheader("Guardar y continuar")

    entries = st.session_state.manual_entries_by_patient[patient_key]
    entries = normalize_entries_to_reference_date(entries, pred_date)
    st.session_state.manual_entries_by_patient[patient_key] = entries

    if not entries:
        st.info("No hay eventos para guardar todavía.")
    else:
        st.markdown(
            """
            <div style="
                padding: 12px 16px;
                border-radius: 12px;
                background: rgba(52, 152, 219, 0.08);
                border: 1px solid rgba(52, 152, 219, 0.25);
                margin-bottom: 14px;
            ">
                Las ingestas y la configuración se guardan en un solo paso.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.button("💾 Guardar eventos", use_container_width=True, type="primary"):
        _, _, _ = save_all_manual_data(
            patient_id=patient_key,
            patient_name=patient_name,
            patient_folder=patient_folder,
            pred_date=pred_date,
            start_time_value=start_time_value,
            glucose_now=glucose_now,
            entries=entries
        )
        st.success("Eventos guardados correctamente.")

    st.markdown("---")

    left, center, right = st.columns([2, 3, 2])

    # with center:
    #     if st.button("📈 Guardar e ir a Pronóstico", use_container_width=True, type="primary"):
    #         _, _, config_payload = save_all_manual_data(
    #             patient_id=patient_key,
    #             patient_name=patient_name,
    #             patient_folder=patient_folder,
    #             pred_date=pred_date,
    #             start_time_value=start_time_value,
    #             glucose_now=glucose_now,
    #             entries=entries
    #         )
    #         st.session_state.manual_prediction_config = config_payload
    #         st.switch_page("pages/4_Pronóstico.py")