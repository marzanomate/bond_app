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

@st.cache_data(show_spinner=False)
def load_all(_cache_buster: int):
    """
    Returns: df_all, df_bonds, df_notes, df_corps, df_mep, mep_al30
    _cache_buster is unused; it only forces cache invalidation periodically.
    """
    data = {}
    for k, url in URLS.items():
        try:
            payload = fetch_json(url)
        except Exception as e:
            st.warning(f"⚠️ Could not fetch `{k}`: {e}")
            payload = []
        df = to_df(payload)
        if k != "mep":
            df["source"] = k
        data[k] = df

    frames = [
        data.get("bonds", pd.DataFrame()),
        data.get("notes", pd.DataFrame()),
        data.get("corps", pd.DataFrame()),
    ]
    df_all = (
        pd.concat(frames, ignore_index=True, sort=False)
        if any(len(f) for f in frames)
        else pd.DataFrame()
    )

    df_mep = data.get("mep", pd.DataFrame()).copy()
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
                mep_al30 = df_mep.loc[mask, bid_col].iloc[0]
            except Exception:
                mep_al30 = None

    return df_all, data.get("bonds"), data.get("notes"), data.get("corps"), df_mep, mep_al30

def filter_df(df: pd.DataFrame, sources: list[str], q: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if "source" in out.columns and sources:
        out = out[out["source"].isin(sources)]

    q = (q or "").strip().lower()
    if q:
        # common text columns; fallback to any object dtype
        cols = [c for c in out.columns if c.lower() in
                ("symbol", "ticker", "desc", "description", "nombre", "isin")]
        if not cols:
            cols = [c for c in out.columns if out[c].dtype == "object"]
        mask = pd.Series(False, index=out.index)
        for c in cols:
            mask |= out[c].astype(str).str.lower().str.contains(q, na=False)
        out = out[mask]

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

# ---------------- UI ----------------
st.title("🇦🇷 ARG Fixed Income — Live Board")

with st.sidebar:
    st.header("Controls")
    sources = st.multiselect(
        "Data sources",
        options=["bonds", "notes", "corps"],
        default=["bonds", "notes", "corps"],
    )
    query = st.text_input("Search (ticker / symbol / ISIN / name)", value="", placeholder="e.g., AL30, GD35, YPF…")
    cache_ttl = st.slider("Auto-refresh every (seconds)", 5, 300, 30,
                          help="Data is re-fetched automatically every N seconds.")
    col1, col2 = st.columns(2)
    with col1:
        refresh = st.button("🔄 Refresh now")
    with col2:
        wide = st.toggle("Wide mode", value=True)
        st.session_state["_wide"] = wide

if st.session_state.get("_wide", True):
    st.set_page_config(layout="wide")

# manual refresh clears cache
if refresh:
    load_all.clear()

# periodic refresh: change cache key every `cache_ttl` seconds
cache_buster = int(time.time() // cache_ttl)

with st.spinner("Fetching live data…"):
    df_all, df_bonds, df_notes, df_corps, df_mep, mep_al30 = load_all(cache_buster)

# KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Rows (All markets)", f"{len(df_all) if df_all is not None else 0:,}")
k2.metric("Rows (MEP)", f"{len(df_mep) if df_mep is not None else 0:,}")
k3.metric("MEP AL30 (Bid)", f"{mep_al30:,.2f}" if isinstance(mep_al30, (int, float)) else "—")
k4.metric("Last refresh (local)", time.strftime("%Y-%m-%d %H:%M:%S"))

st.divider()

tab_all, tab_bonds, tab_notes, tab_corps, tab_mep = st.tabs(
    ["Overview (All)", "Bonds", "Notes", "Corps", "MEP"]
)

with tab_all:
    df_f = filter_df(df_all, sources, query)
    st.caption(f"Showing {len(df_f):,} rows")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Download filtered (CSV)", "arg_all_filtered.csv")

with tab_bonds:
    df_f = filter_df(df_bonds, ["bonds"] if "bonds" in sources else [], query)
    st.caption(f"Showing {len(df_f):,} rows")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Download bonds (CSV)", "arg_bonds.csv")

with tab_notes:
    df_f = filter_df(df_notes, ["notes"] if "notes" in sources else [], query)
    st.caption(f"Showing {len(df_f):,} rows")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Download notes (CSV)", "arg_notes.csv")

with tab_corps:
    df_f = filter_df(df_corps, ["corps"] if "corps" in sources else [], query)
    st.caption(f"Showing {len(df_f):,} rows")
    st.dataframe(df_f, use_container_width=True, hide_index=True)
    make_download(df_f, "⬇️ Download corps (CSV)", "arg_corps.csv")

with tab_mep:
    st.subheader("Raw MEP")
    st.caption(f"Rows: {len(df_mep) if df_mep is not None else 0:,}")
    st.dataframe(df_mep, use_container_width=True, hide_index=True)
    make_download(df_mep, "⬇️ Download mep (CSV)", "arg_mep.csv")

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
