# app.py
import json
import time
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="ARG Fixed Income Live", layout="wide")

URLS = {
    "bonds": "https://data912.com/live/arg_bonds",
    "notes": "https://data912.com/live/arg_notes",
    "corps": "https://data912.com/live/arg_corp",
    "mep":   "https://data912.com/live/mep",
}

# UI labels you asked for
LABELS = {"bonds": "Bonos", "notes": "Letras", "corps": "Obligaciones Negociables"}

# Columns you want, in this order
TARGET_ORDER = [
    "Ticker",
    "Cantidad (BID)",
    "Precio (BID)",
    "Cantidad (ASK)",
    "Precio (ASK)",
    "Volumen",
    "Cantidad (Operaciones)",
    "Último",
    "Cambio de Precio",
    "Tipo",
]

def to_df(payload: dict | list) -> pd.DataFrame:
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        payload = [payload]
    try:
        return pd.json_normalize(payload)
    except Exception:
        return pd.DataFrame(payload)

def fetch_json(url: str, timeout: int = 20) -> dict | list:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        return json.loads(r.text)

def coalesce_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
        # also try case-insensitive
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

def normalize_view(df_in: pd.DataFrame, tipo_label: str) -> pd.DataFrame:
    """
    Map a raw API dataframe to the requested 10 columns.
    Tries multiple common field names to be robust.
    """
    if df_in is None or df_in.empty:
        out = pd.DataFrame(columns=TARGET_ORDER)
        return out

    df = df_in.copy()

    # Candidate names for each target field (try in order)
    ticker_cands = ["ticker", "symbol", "isin", "nombre", "name", "desc", "description"]
    bid_qty_cands = ["bid_qty", "bidQty", "bidQuantity", "cantidadBid", "cant_bid", "qty_bid", "qtyBid"]
    bid_px_cands  = ["bid", "precioCompra", "bidPrice", "priceBid", "px_bid", "bestBid"]
    ask_qty_cands = ["ask_qty", "askQty", "askQuantity", "cantidadAsk", "cant_ask", "qty_ask", "qtyAsk"]
    ask_px_cands  = ["ask", "precioVenta", "askPrice", "priceAsk", "px_ask", "bestAsk"]
    vol_cands     = ["volume", "vol", "volumen", "turnover"]
    trades_cands  = ["trades", "operations", "cantidadOperaciones", "ops", "count", "numTrades"]
    last_cands    = ["last", "ultimo", "lastPrice", "px_last", "price", "ltp"]
    chg_cands     = ["change", "chg", "cambio", "pct_change", "changePct", "pctChange", "var", "variacion"]

    # Resolve actual columns present
    col_map = {
        "Ticker": coalesce_col(df, ticker_cands),
        "Cantidad (BID)": coalesce_col(df, bid_qty_cands),
        "Precio (BID)": coalesce_col(df, bid_px_cands),
        "Cantidad (ASK)": coalesce_col(df, ask_qty_cands),
        "Precio (ASK)": coalesce_col(df, ask_px_cands),
        "Volumen": coalesce_col(df, vol_cands),
        "Cantidad (Operaciones)": coalesce_col(df, trades_cands),
        "Último": coalesce_col(df, last_cands),
        "Cambio de Precio": coalesce_col(df, chg_cands),
    }

    # Build output with safe defaults
    out = pd.DataFrame()
    for tgt, src in col_map.items():
        if src is not None:
            out[tgt] = df[src]
        else:
            # create empty if missing
            out[tgt] = pd.Series([None] * len(df))

    # Add Tipo label
    out["Tipo"] = tipo_label

    # Numeric clean-up on price/qty-like columns (best effort)
    numeric_cols = [
        "Cantidad (BID)", "Precio (BID)", "Cantidad (ASK)", "Precio (ASK)",
        "Volumen", "Cantidad (Operaciones)", "Último", "Cambio de Precio"
    ]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    # Ensure order
    out = out[TARGET_ORDER]

    # Optional: Ticker as string
    out["Ticker"] = out["Ticker"].astype(str)

    return out

@st.cache_data(show_spinner=False)
def load_all(_cache_buster: int):
    data = {}
    for k, url in URLS.items():
        try:
            payload = fetch_json(url)
        except Exception as e:
            st.warning(f"⚠️ No se pudo obtener `{k}`: {e}")
            payload = []
        df = to_df(payload)
        data[k] = df

    # Normalize each into your display schema + tag "Tipo"
    bonds_view = normalize_view(data.get("bonds"), LABELS["bonds"])
    notes_view = normalize_view(data.get("notes"), LABELS["notes"])
    corps_view = normalize_view(data.get("corps"), LABELS["corps"])

    # Combined view
    df_all = pd.concat([bonds_view, notes_view, corps_view], ignore_index=True)

    # MEP raw (left as-is, separate tab)
    df_mep = data.get("mep", pd.DataFrame()).copy()

    # Try to pull AL30 MEP bid
    mep_al30 = None
    if not df_mep.empty:
        ticker_col = "ticker" if "ticker" in df_mep.columns else (
            "symbol" if "symbol" in df_mep.columns else None
        )
        bid_col = "bid" if "bid" in df_mep.columns else (
            "precioCompra" if "precioCompra" in df_mep.columns else None
        )
        if ticker_col and bid_col:
            try:
                mask = df_mep[ticker_col].astype(str).str.upper().eq("AL30")
                mep_al30 = pd.to_numeric(df_mep.loc[mask, bid_col], errors="coerce").dropna().iloc[0]
            except Exception:
                mep_al30 = None

    return df_all, bonds_view, notes_view, corps_view, df_mep, mep_al30

def filter_df(df: pd.DataFrame, tipos: list[str], q: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=TARGET_ORDER)
    out = df.copy()

    # Filter by Tipo (Bonos/Letras/ON)
    if tipos:
        out = out[out["Tipo"].isin(tipos)]

    # Text query over Ticker
    q = (q or "").strip().lower()
    if q:
        mask = out["Ticker"].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

    # Keep only target columns, in order
    missing = [c for c in TARGET_ORDER if c not in out.columns]
    for c in missing:
        out[c] = None
    out = out[TARGET_ORDER]

    return out

def make_download(df: pd.DataFrame, label: str, filename: str):
    if df is None or df.empty:
        st.download_button(label, data=b"", file_name=filename, disabled=True)
    else:
        st.download_button(
            label,
            df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv"
        )

# ============== UI ==============
st.title("🇦🇷 ARG Fixed Income — Live Board")

with st.sidebar:
    st.header("Controles")
    # User-facing options renamed
    opciones = [LABELS["bonds"], LABELS["notes"], LABELS["corps"]]
    tipos_sel = st.multiselect(
        "Tipo",
        options=opciones,
        default=opciones
    )
    query = st.text_input("Buscar (Ticker)", value="", placeholder="Ej.: AL30, GD35, YPFD…")
    cache_ttl = st.slider("Auto-refrescar cada (segundos)", 5, 300, 30)
    col1, col2 = st.columns(2)
    with col1:
        refresh = st.button("🔄 Refrescar ahora")
    with col2:
        wide = st.toggle("Modo ancho", value=True)
        st.session_state["_wide"] = wide

if st.session_state.get("_wide", True):
    st.set_page_config(layout="wide")

if refresh:
    load_all.clear()

cache_buster = int(time.time() // cache_ttl)

with st.spinner("Cargando datos en vivo…"):
    df_all, df_bonos, df_letras, df_ons, df_mep, mep_al30 = load_all(cache_buster)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Filas (Bonos/Letras/ON)", f"{len(df_all):,}")
k2.metric("Filas (MEP)", f"{len(df_mep) if df_mep is not None else 0:,}")
k3.metric("MEP AL30 (Bid)", f"{mep_al30:,.2f}" if isinstance(mep_al30, (int, float)) else "—")
k4.metric("Última actualización", time.strftime("%Y-%m-%d %H:%M:%S"))

st.divider()

tab_all, tab_bonos, tab_letras, tab_ons, tab_mep = st.tabs(
    ["Resumen (Todos)", "Bonos", "Letras", "Obligaciones Negociables", "MEP"]
)

with tab_all:
    df_f = filter_df(df_all, tipos_sel, query)
    st.caption(f"Mostrando {len(df_f):,} filas")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Descargar (CSV)", "arg_todos.csv")

with tab_bonos:
    df_f = filter_df(df_bonos, [LABELS["bonds"]] if LABELS["bonds"] in tipos_sel else [], query)
    st.caption(f"Mostrando {len(df_f):,} filas")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Descargar Bonos (CSV)", "arg_bonos.csv")

with tab_letras:
    df_f = filter_df(df_letras, [LABELS["notes"]] if LABELS["notes"] in tipos_sel else [], query)
    st.caption(f"Mostrando {len(df_f):,} filas")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Descargar Letras (CSV)", "arg_letras.csv")

with tab_ons:
    df_f = filter_df(df_ons, [LABELS["corps"]] if LABELS["corps"] in tipos_sel else [], query)
    st.caption(f"Mostrando {len(df_f):,} filas")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Descargar ON (CSV)", "arg_on.csv")

with tab_mep:
    st.subheader("MEP (raw)")
    st.caption(f"Filas: {len(df_mep) if df_mep is not None else 0:,}")
    st.dataframe(df_mep, use_container_width=True, hide_index=True)
    make_download(df_mep, "⬇️ Descargar MEP (CSV)", "arg_mep.csv")

    ticker_col = None
    if df_mep is not None and not df_mep.empty:
        ticker_col = "ticker" if "ticker" in df_mep.columns else (
            "symbol" if "symbol" in df_mep.columns else None
        )
    if ticker_col:
        al30 = df_mep[df_mep[ticker_col].astype(str).str.upper().eq("AL30")]
        if not al30.empty:
            st.markdown("**AL30 (MEP) snapshot**")
            st.dataframe(al30, use_container_width=True, hide_index=True)
