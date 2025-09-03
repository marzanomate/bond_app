

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(
    page_title="ONs — Fundamentals",
    page_icon="💼",
    layout="wide"
)
st.title("💼 ONs — Fundamentals")

with st.sidebar:
    st.header("⚙️ Configuración")

    # Excel path (default to your local path)
    default_excel_path = r"C:\Users\mmarzano\Documents\Modelos - Calculadora\listado_ons.xlsx"
    excel_path = st.text_input("Ruta del Excel de ONs", value=default_excel_path)

    prefer_col = st.selectbox("Precio preferido", options=["px_bid", "px_ask"], index=0)

    st.caption("Opcional: cargar un CSV/Parquet con `df_all` si no querés usar los endpoints.")
    up_df_all = st.file_uploader("Subí df_all (CSV o Parquet)", type=["csv", "parquet"])

    do_fetch = st.checkbox("Intentar traer precios de data912 (bonds/notes/corps)", value=True)
    reload_btn = st.button("🔄 Recargar datos")

# -------------------------
# Data loading helpers (cached)
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_df_all_from_endpoints():
    import requests
    def fetch_json(url):
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    def to_df(payload):
        if isinstance(payload, dict):
            for key in ("data", "results", "items", "bonds", "notes"):
                if key in payload and isinstance(payload[key], list):
                    payload = payload[key]
                    break
        return pd.json_normalize(payload)

    url_bonds = "https://data912.com/live/arg_bonds"
    url_notes = "https://data912.com/live/arg_notes"
    url_corps = "https://data912.com/live/arg_corp"

    data_bonds = fetch_json(url_bonds)
    data_notes = fetch_json(url_notes)
    data_corps = fetch_json(url_corps)

    df_bonds = to_df(data_bonds)
    df_notes = to_df(data_notes)
    df_corps = to_df(data_corps)
    df_bonds["source"] = "bonds"
    df_notes["source"] = "notes"
    df_corps["source"] = "corps"

    df_all = pd.concat([df_bonds, df_notes, df_corps], ignore_index=True, sort=False)

    # Try to standardize columns commonly seen in data912 payloads
    # Ensure we have at least: symbol, px_bid, px_ask
    # If the payload names differ, try to create best-effort mappings.
    if "symbol" not in df_all.columns and "ticker" in df_all.columns:
        df_all = df_all.rename(columns={"ticker": "symbol"})
    # Guess bid/ask names if needed
    guess_map = {
        "bid": "px_bid", "ask": "px_ask",
        "px_bid_": "px_bid", "px_ask_": "px_ask",
        "price_bid": "px_bid", "price_ask": "px_ask"
    }
    for c_old, c_new in guess_map.items():
        if c_old in df_all.columns and c_new not in df_all.columns:
            df_all = df_all.rename(columns={c_old: c_new})

    return df_all

@st.cache_data(show_spinner=False)
def load_everything(excel_path, df_all, prefer_col):
    bonds = load_ons_from_excel(excel_path, df_all, price_col_prefer=prefer_col)
    df_metrics = bond_fundamentals_ons(bonds)
    return df_metrics

# -------------------------
# Build df_all
# -------------------------
df_all = None
status_msgs = []

if up_df_all is not None:
    try:
        if up_df_all.name.lower().endswith(".csv"):
            df_all = pd.read_csv(up_df_all)
        else:
            df_all = pd.read_parquet(up_df_all)
        status_msgs.append(f"📁 `df_all` cargado desde archivo subido ({len(df_all):,} filas).")
    except Exception as e:
        st.warning(f"⚠️ No pude leer el archivo subido como df_all: {e}")

if df_all is None and do_fetch:
    with st.spinner("Descargando precios de data912..."):
        try:
            df_all = fetch_df_all_from_endpoints()
            status_msgs.append(f"🌐 `df_all` descargado de endpoints ({len(df_all):,} filas).")
        except Exception as e:
            st.warning(f"⚠️ No pude traer data de data912: {e}")

# If still None, create an empty df_all to avoid crash, but warn
if df_all is None:
    st.error("❌ No hay `df_all`. Subí un archivo con columnas al menos ['symbol','px_bid'] o habilitá el fetch.")
    st.stop()

# -------------------------
# Build df_metrics
# -------------------------
if reload_btn:
    # clear caches when user clicks reload
    fetch_df_all_from_endpoints.clear()
    load_everything.clear()

with st.spinner("Calculando métricas..."):
    try:
        df_metrics = load_everything(excel_path, df_all, prefer_col)
        status_msgs.append(f"✅ Métricas calculadas: {len(df_metrics):,} ONs.")
    except Exception as e:
        st.error(f"❌ Error al calcular métricas: {e}")
        st.stop()

# -------------------------
# Sidebar status
# -------------------------
with st.sidebar:
    st.divider()
    for msg in status_msgs:
        st.write(msg)
    st.caption("Tip: cambiá el precio preferido (bid/ask) o recargá datos si actualizaste el Excel.")

# -------------------------
# Filters
# -------------------------
st.subheader("Filtros")
empresas = sorted([e for e in df_metrics["Empresa"].dropna().unique()])
sel_empresas = st.multiselect("Empresa", empresas, default=empresas)

df_view = df_metrics[df_metrics["Empresa"].isin(sel_empresas)].reset_index(drop=True)

# -------------------------
# Display
# -------------------------
st.subheader("Tabla de métricas")

# Column formats for nicer display
col_config = {
    "Cupón": st.column_config.NumberColumn("Cupón (%)", help="Tasa nominal anual", format="%.4f"),
    "Precio": st.column_config.NumberColumn("Precio", format="%.2f"),
    "Yield": st.column_config.NumberColumn("Yield (%)", format="%.2f"),
    "TNA_180": st.column_config.NumberColumn("TNA 180 (%)", format="%.2f"),
    "Dur": st.column_config.NumberColumn("Duración (años)", format="%.2f"),
    "MD": st.column_config.NumberColumn("Mod. Duration (años)", format="%.2f"),
    "Conv": st.column_config.NumberColumn("Convexidad", format="%.2f"),
    "Current Yield": st.column_config.NumberColumn("Current Yield (%)", format="%.2f"),
    "Paridad (%)": st.column_config.NumberColumn("Paridad (%)", format="%.2f"),
}

st.dataframe(
    df_view,
    use_container_width=True,
    hide_index=True,
    column_config=col_config
)

# -------------------------
# Downloads
# -------------------------
c1, c2 = st.columns(2)
with c1:
    csv = df_view.to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Descargar CSV (filtrado)", data=csv, file_name="ons_fundamentals.csv", mime="text/csv")
with c2:
    try:
        import io
        import xlsxwriter
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_view.to_excel(writer, index=False, sheet_name="Fundamentals")
        st.download_button("⬇️ Descargar Excel (filtrado)", data=output.getvalue(),
                           file_name="ons_fundamentals.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    except Exception:
        st.caption("Para exportar a Excel, instalá `xlsxwriter`.")
