import streamlit as st
import pandas as pd
from pathlib import Path

from kidia.state import init_state
from kidia.auth import logout

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="KIDIA – Dashboard", layout="wide")
init_state()

# 🔐 Seguridad
if not st.session_state.get("authenticated", False):
    st.switch_page("app.py")

from kidia.ui import render_kidia_header
render_kidia_header()

st.markdown("""
<style>
    .patients-title {
        margin-top: 0rem !important;
        margin-bottom: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# =========================
# CSS DE TABS
# =========================
st.markdown("""
<style>
    button[data-baseweb="tab"] p {
        font-size: 1.15rem !important;
        font-weight: 800 !important;
    }

    button[data-baseweb="tab"] {
        padding: 14px 24px !important;
        height: auto !important;
        flex: 1 1 0% !important;
        justify-content: center !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        border-bottom: 3px solid #2563EB !important;
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
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR LIMPIO
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
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-bottom-space"></div>', unsafe_allow_html=True)

    if "confirm_logout" not in st.session_state:
        st.session_state.confirm_logout = False

    if not st.session_state.confirm_logout:
        if st.button("Cerrar sesión", use_container_width=True):
            st.session_state.confirm_logout = True
            st.rerun()
    else:
        st.warning("¿Seguro que deseas cerrar sesión?")

        c1, c2 = st.columns(2)

        with c1:
            if st.button("Sí, salir", use_container_width=True):
                st.session_state.confirm_logout = False
                logout()
                st.switch_page("app.py")

        with c2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state.confirm_logout = False
                st.rerun()

# =========================
# RUTAS / PERSISTENCIA
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parents[1]
except Exception:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS_CSV = DATA_DIR / "patients.csv"
REQUIRED_COLS = ["ID", "Nombre"]

# =========================
# HELPERS
# =========================
def _ensure_patients_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(columns=REQUIRED_COLS)

    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = ""

    df["ID"] = df["ID"].astype(str)
    df["Nombre"] = df["Nombre"].astype(str)

    return df[REQUIRED_COLS].copy()


def load_patients_from_disk() -> pd.DataFrame:
    if PATIENTS_CSV.exists():
        try:
            df = pd.read_csv(PATIENTS_CSV, dtype=str)
            return _ensure_patients_df(df)
        except Exception:
            return pd.DataFrame(columns=REQUIRED_COLS)
    return pd.DataFrame(columns=REQUIRED_COLS)


def save_patients_to_disk(df: pd.DataFrame) -> None:
    df = _ensure_patients_df(df)
    df.to_csv(PATIENTS_CSV, index=False)


def _clear_active_patient():
    st.session_state.active_patient = None
    st.session_state.patient_id = None
    st.session_state.patient_info = None


# =========================
# SINCRONIZAR STATE <-> DISCO
# =========================
st.session_state.patients = _ensure_patients_df(st.session_state.get("patients"))

if st.session_state.patients.empty and PATIENTS_CSV.exists():
    st.session_state.patients = load_patients_from_disk()

save_patients_to_disk(st.session_state.patients)

patients_df = _ensure_patients_df(st.session_state.get("patients"))

# =========================
# UI PRINCIPAL
# =========================
st.markdown("""
<h2 class="patients-title" style='text-align: center;'>
    👶 Gestión de Pacientes
</h2>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs([
    "📋 Consultar",
    "✏️ Editar"
])

# -------------------------------------------------
# TAB 1: CONSULTAR
# -------------------------------------------------
with tab1:
    patients_df = _ensure_patients_df(st.session_state.patients)

    st.markdown("### 📋 Lista de pacientes")

    if patients_df.empty:
        st.info("No hay pacientes registrados.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            use_id_filter = st.checkbox("Buscar por ID", key="use_id_filter")
            search_id = st.text_input(
                "ID del paciente",
                placeholder="Ejemplo: CHILD_001",
                key="search_patient_id",
                disabled=not use_id_filter
            )

        with c2:
            use_name_filter = st.checkbox("Buscar por Nombre", key="use_name_filter")
            search_name = st.text_input(
                "Nombre del paciente",
                placeholder="Ejemplo: Niño 1",
                key="search_patient_name",
                disabled=not use_name_filter
            )

        df_filtered = patients_df.copy()

        if use_id_filter and search_id.strip():
            term_id = search_id.strip().lower()
            df_filtered = df_filtered[
                df_filtered["ID"].astype(str).str.lower().str.contains(term_id, na=False)
            ]

        if use_name_filter and search_name.strip():
            term_name = search_name.strip().lower()
            df_filtered = df_filtered[
                df_filtered["Nombre"].astype(str).str.lower().str.contains(term_name, na=False)
            ]

        df_filtered = df_filtered.reset_index(drop=True)

        st.caption(f"Pacientes encontrados: {len(df_filtered)}")

        if df_filtered.empty:
            st.warning("No se encontraron pacientes con los filtros seleccionados.")
        else:
            st.dataframe(
                df_filtered,
                use_container_width=True,
                hide_index=True
            )

# -------------------------------------------------
# TAB 2: EDITAR
# -------------------------------------------------
with tab2:
    patients_df = _ensure_patients_df(st.session_state.patients)

    st.markdown("### ✏️ Editar paciente")

    if patients_df.empty:
        st.info("No hay pacientes registrados para editar.")
    else:
        c1, c2 = st.columns(2)

        with c1:
            use_edit_id_filter = st.checkbox("Buscar por ID", key="use_edit_id_filter")
            search_edit_id = st.text_input(
                "Filtrar por ID",
                placeholder="Ejemplo: CHILD_001",
                key="search_edit_id",
                disabled=not use_edit_id_filter
            )

        with c2:
            use_edit_name_filter = st.checkbox("Buscar por Nombre", key="use_edit_name_filter")
            search_edit_name = st.text_input(
                "Filtrar por Nombre",
                placeholder="Ejemplo: Niño 1",
                key="search_edit_name",
                disabled=not use_edit_name_filter
            )

        df_filtered = patients_df.copy()

        if use_edit_id_filter and search_edit_id.strip():
            term_id = search_edit_id.strip().lower()
            df_filtered = df_filtered[
                df_filtered["ID"].astype(str).str.lower().str.contains(term_id, na=False)
            ]

        if use_edit_name_filter and search_edit_name.strip():
            term_name = search_edit_name.strip().lower()
            df_filtered = df_filtered[
                df_filtered["Nombre"].astype(str).str.lower().str.contains(term_name, na=False)
            ]

        df_filtered = df_filtered.reset_index(drop=True)

        st.caption(f"Pacientes encontrados: {len(df_filtered)}")

        if df_filtered.empty:
            st.warning("No se encontraron pacientes con los filtros seleccionados.")
        else:
            selected_id = st.selectbox(
                "Selecciona el paciente por ID",
                df_filtered["ID"].tolist(),
                key="edit_patient_id"
            )

            row = patients_df.loc[patients_df["ID"] == str(selected_id)]
            current_name = row["Nombre"].iloc[0] if not row.empty else ""

            new_name = st.text_input(
                "Nuevo nombre",
                value=current_name,
                key="edit_patient_name"
            )

            if st.button("Guardar cambios", type="primary", use_container_width=True):
                new_name = (new_name or "").strip()

                if not new_name:
                    st.error("El nombre no puede quedar vacío.")
                else:
                    idx = patients_df.index[patients_df["ID"] == str(selected_id)]

                    if len(idx) == 0:
                        st.error("No se encontró el paciente seleccionado.")
                    else:
                        patients_df.loc[idx[0], "Nombre"] = new_name
                        st.session_state.patients = patients_df.reset_index(drop=True)
                        save_patients_to_disk(st.session_state.patients)

                        active_patient = st.session_state.get("active_patient")
                        if active_patient and str(active_patient.get("ID")) == str(selected_id):
                            st.session_state.active_patient = {
                                "ID": str(selected_id),
                                "Nombre": new_name
                            }
                            st.session_state.patient_id = str(selected_id)
                            st.session_state.patient_info = {
                                "ID": str(selected_id),
                                "Nombre": new_name
                            }

                        st.success("Paciente actualizado correctamente")
                        st.rerun()