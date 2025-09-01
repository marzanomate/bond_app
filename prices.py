# app.py
import json, time
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import streamlit as st
import scipy.optimize as opt

import alphacast as Alphacast

import QuantLib as ql  # quantlib-python

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="ARG FI Metrics", layout="wide")

URLS = {
    "bonds": "https://data912.com/live/arg_bonds",
    "notes": "https://data912.com/live/arg_notes",
    "corps": "https://data912.com/live/arg_corp",
    "mep":   "https://data912.com/live/mep",
}

# -----------------------------
# HELPERS (requests/json -> df)
# -----------------------------
def fetch_json(url: str, timeout: int = 20):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    try:
        return r.json()
    except json.JSONDecodeError:
        return json.loads(r.text)

def to_df(payload):
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    if not isinstance(payload, (list, tuple)):
        payload = [payload]
    return pd.json_normalize(payload)

@st.cache_data(show_spinner=False, ttl=60)
def load_market():
    data = {}
    for k, url in URLS.items():
        try:
            data[k] = to_df(fetch_json(url))
        except Exception as e:
            st.warning(f"⚠️ No se pudo obtener `{k}`: {e}")
            data[k] = pd.DataFrame()

    # Combined prices without showing it in UI
    frames = [data.get("bonds"), data.get("notes"), data.get("corps")]
    df_all = pd.concat([f for f in frames if f is not None], ignore_index=True, sort=False)

    # MEP AL30 bid
    df_mep = data.get("mep", pd.DataFrame())
    mep = None
    if not df_mep.empty:
        tcol = "ticker" if "ticker" in df_mep.columns else ("symbol" if "symbol" in df_mep.columns else None)
        bcol = "bid" if "bid" in df_mep.columns else ("precioCompra" if "precioCompra" in df_mep.columns else None)
        if tcol and bcol:
            m = df_mep[tcol].astype(str).str.upper().eq("AL30")
            if m.any():
                mep = pd.to_numeric(df_mep.loc[m, bcol], errors="coerce").dropna().max()

    return df_all, df_mep, mep

# -----------------------------
# YOUR DATA PREP (Alphacast -> CER/TAMAR)
# -----------------------------
def prep_cer_tamar(df_alphacast: pd.DataFrame):
    df = df_alphacast.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

    df_cer = (df.loc[df["Category 2"].eq("CER"), ["Date","Value"]]
                .sort_values("Date").set_index("Date")
                .rename(columns={"Value":"CER"}))
    df_tamar = (df.loc[df["Category 2"].eq("TAMAR (en % n.a)"), ["Date","Value"]]
                  .sort_values("Date").set_index("Date")
                  .rename(columns={"Value":"TAMAR_pct_na"}))

    df_cer = df_cer[~df_cer.index.duplicated(keep="last")]
    df_tamar = df_tamar[~df_tamar.index.duplicated(keep="last")]
    return df_cer, df_tamar

@st.cache_data(show_spinner=False, ttl=180)
def load_alphacast_and_metrics():
    # --- fake a minimal Alphacast fetch path hook: user is already passing df_alphacast normally ---
    # If you actually want to call Alphacast SDK, plug it here.
    # For now assume user brings df_alphacast externally or you can stop here and just return None.
    return None  # We’ll compute CER/TAMAR via endpoints or the user can paste df_alphacast.

# -----------------------------
# CLASSES (shortened, minimal fixes)
# -----------------------------
class lecaps:
    def __init__(self, name, start_date, end_date, tem, price):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.tem = float(tem)
        self.price = float(price)
        self.settlement = datetime.today() + timedelta(days=1)
        self.calendar = ql.Argentina(ql.Argentina.Merval)
        self.convention = ql.Following

    def _adj(self, dt):
        qd = ql.Date(dt.day, dt.month, dt.year)
        ad = self.calendar.adjust(qd, self.convention)
        return datetime(ad.year(), int(ad.month()), ad.dayOfMonth())

    def generate_payment_dates(self):
        return [self._adj(self.settlement).strftime("%Y-%m-%d"),
                self._adj(self.end_date).strftime("%Y-%m-%d")]

    def cash_flow(self):
        dc = ql.Thirty360(ql.Thirty360.BondBasis)
        ql_start = ql.Date(self.start_date.day, self.start_date.month, self.start_date.year)
        ql_end   = ql.Date(self.end_date.day,   self.end_date.month,   self.end_date.year)
        days = dc.dayCount(ql_start, ql_end)
        months = days/30
        growth = (1 + self.tem) ** months - 1
        final = 100 * (1 + growth)
        return [-self.price, final]

    def xnpv(self, rate):
        d0 = self.settlement
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        cfs = self.cash_flow()
        return sum(cf / (1 + rate) ** ((dt - d0).days/365.0) for cf, dt in zip(cfs, dates))

    def xirr(self):
        f = lambda r: self.xnpv(r)
        try:
            r = opt.newton(f, 0.0)
        except RuntimeError:
            r = opt.brentq(f, -0.99, 10.0)
        return r*100

    def tem_from_irr(self):
        irr = self.xirr()/100
        return ((1+irr)**(30/365)-1)*100

    def tna30(self):
        tem = self.tem_from_irr()/100
        return tem*12*100

    def duration(self):
        irr = self.xirr()/100
        d0 = self.settlement
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        cfs = self.cash_flow()
        num = sum(((dt-d0).days/365.0)*cf/(1+irr)**(((dt-d0).days/365.0)) for cf, dt in zip(cfs[1:], dates[1:]))
        pv  = sum(cf/(1+irr)**(((dt-d0).days/365.0)) for cf, dt in zip(cfs[1:], dates[1:]))
        return num/pv if pv else np.nan

    def modified_duration(self):
        irr = self.xirr()/100
        return self.duration()/(1+irr)

class dlk(lecaps):
    def __init__(self, name, start_date, end_date, mep, price):
        super().__init__(name, start_date, end_date, tem=0.0, price=price)
        self.mep = float(mep)

    def cash_flow(self):
        return [-self.price, 100*self.mep]

class cer_bonos:
    def __init__(self, name, cer_final, cer_inicial, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price, fr):
        self.name = name
        self.cer_inicial = float(cer_inicial)
        self.cer_final   = float(cer_final)
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = int(payment_frequency)
        self.amortization_dates = amortization_dates
        self.amortizations = amortizations
        self.rate = float(rate)     # % nominal anual
        self.price = float(price)
        self.frequency = int(fr)    # coupons per year

    def generate_payment_dates(self):
        dates = []
        settlement = datetime.today() + timedelta(days=1)
        dates.append(settlement.strftime("%Y-%m-%d"))
        current = self.start_date
        while current <= self.end_date:
            if current > settlement:
                dates.append(current.strftime("%Y-%m-%d"))
            current = current + relativedelta(months=self.payment_frequency)
        return dates

    def residual_value(self):
        adj = self.cer_final/self.cer_inicial
        dates = self.generate_payment_dates()
        residual = 100*adj
        res = []
        for d in dates:
            res.append(residual)
            if d in self.amortization_dates:
                residual -= self.amortizations[self.amortization_dates.index(d)]*adj
        return res

    def coupon_payments(self):
        cpns, res = [], self.residual_value()
        for i,_ in enumerate(self.generate_payment_dates()):
            cpns.append(0 if i==0 else (self.rate/100/self.frequency)*res[i-1])
        return cpns

    def amortization_payments(self):
        adj = self.cer_final/self.cer_inicial
        dates = self.generate_payment_dates()
        caps = [0.0]
        amap = dict(zip(self.amortization_dates, self.amortizations))
        for d in dates[1:]:
            caps.append(amap.get(d, 0.0)*adj)
        return caps

    def cash_flow(self):
        caps = self.amortization_payments()
        cpns = self.coupon_payments()
        cfs = [-self.price]
        cfs += [c+a for c,a in zip(cpns[1:], caps[1:])]
        return cfs

    def xnpv(self, rate):
        d0 = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        cfs = self.cash_flow()
        return sum(cf/(1+rate)**(((dt-d0).days/365.0)) for cf, dt in zip(cfs, dates))

    def xirr(self):
        f = lambda r: self.xnpv(r)
        try:
            r = opt.newton(f, 0.0)
        except RuntimeError:
            r = opt.brentq(f, -0.99, 10.0)
        return r*100

    def tna30(self):
        irr = self.xirr()/100
        tem = (1+irr)**(30/365)-1
        return tem*12*100

    def duration(self):
        irr = self.xirr()/100
        d0  = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        cfs = self.cash_flow()
        flows = list(zip(cfs[1:], dates[1:]))
        pv = sum(cf/(1+irr)**(((dt-d0).days/365.0)) for cf, dt in flows)
        num = sum(((dt-d0).days/365.0)*cf/(1+irr)**(((dt-d0).days/365.0)) for cf, dt in flows)
        return num/pv if pv else np.nan

    def modified_duration(self):
        irr = self.xirr()/100
        d = self.duration()
        return d/(1+irr) if d==d else np.nan

    def convexity(self):
        irr = self.xirr()/100
        d0  = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        cfs = self.cash_flow()
        flows = list(zip(cfs[1:], dates[1:]))
        pv = sum(cf/(1+irr)**(((dt-d0).days/365.0)) for cf, dt in flows)
        if pv == 0: return np.nan
        cx = sum(cf * t * (t+1) / (1+irr)**(t+2)
                 for cf, t in [(cf, (dt-d0).days/365.0) for cf, dt in flows]) / pv
        return cx

    def current_yield(self):
        cpns = self.coupon_payments()
        d0 = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(d, "%Y-%m-%d") for d in self.generate_payment_dates()]
        fut = [i for i,dt in enumerate(dates) if dt>d0 and cpns[i]>0]
        if not fut: return np.nan
        i0 = fut[0]
        n = min(self.frequency, len(cpns)-i0)
        ann = sum(cpns[i0:i0+n])
        return ann/self.price*100

# -----------------------------
# LOAD MARKET
# -----------------------------
df_all, df_mep, mep_al30 = load_market()

# Ticker -> px_bid helper (don’t show df_all)
def px_lookup(tk):
    if df_all is None or df_all.empty:
        return np.nan
    if "symbol" in df_all.columns and "px_bid" in df_all.columns:
        s = df_all.loc[df_all["symbol"].astype(str).str.upper()==str(tk).upper(), "px_bid"]
        if not s.empty:
            return float(pd.to_numeric(s, errors="coerce").dropna().head(1).values[0])
    return np.nan

# -----------------------------
# DEFINE SAMPLE INSTRUMENTS (your earlier ones)
# -----------------------------
# Minimal CER anchor (user previously computes cer_final)
# If you already computed cer_final in your workflow, set it here.
cer_final = 500.0  # <--- replace with your computed value
def make_tx25(price=None):
    return cer_bonos(
        name="TX25",
        cer_final=cer_final, cer_inicial=46.20846297,
        start_date=datetime(2022,11,9), end_date=datetime(2025,11,9),
        payment_frequency=6,
        amortization_dates=["2025-11-09"], amortizations=[100],
        rate=1.8, price=price if price is not None else px_lookup("TX25"), fr=2
    )

def make_tx26(price=None):
    return cer_bonos(
        name="TX26",
        cer_final=cer_final, cer_inicial=22.5439510895903,
        start_date=datetime(2020,11,9), end_date=datetime(2026,11,9),
        payment_frequency=6,
        amortization_dates=["2024-11-09","2025-05-09","2025-11-09","2026-05-09","2026-11-09"],
        amortizations=[20,20,20,20,20],
        rate=2.0, price=price if price is not None else px_lookup("TX26"), fr=2
    )

def make_tx28(price=None):
    return cer_bonos(
        name="TX28",
        cer_final=cer_final, cer_inicial=22.5439510895903,
        start_date=datetime(2020,11,9), end_date=datetime(2028,11,9),
        payment_frequency=6,
        amortization_dates=[
            "2024-05-09","2024-11-09","2025-05-09","2025-11-09","2026-05-09",
            "2026-11-09","2027-05-09","2027-11-09","2028-05-09","2028-11-09"],
        amortizations=[10]*10,
        rate=2.25, price=price if price is not None else px_lookup("TX28"), fr=2
    )

def make_dicp(price=None):
    return cer_bonos(
        name="DICP",
        cer_final=cer_final*(1+0.26994), cer_inicial=1.45517953387336,
        start_date=datetime(2003,12,31), end_date=datetime(2033,12,31),
        payment_frequency=6,
        amortization_dates=[f"{y}-{m:02d}-30" if m in (6,) else f"{y}-12-31"
                            for y in range(2024,2034) for m in (6,12)],
        amortizations=[5]*20, rate=5.83,
        price=price if price is not None else px_lookup("DICP"), fr=2
    )

def make_cuap(price=None):
    return cer_bonos(
        name="CUAP",
        cer_final=cer_final*(1+0.388667433600987), cer_inicial=1.45517953387336,
        start_date=datetime(2003,12,31), end_date=datetime(2045,12,31),
        payment_frequency=6,
        amortization_dates=[f"{y}-{m:02d}-30" if m in (6,) else f"{y}-12-31"
                            for y in range(2036,2046) for m in (6,12)],
        amortizations=[5]*20, rate=3.31,
        price=price if price is not None else px_lookup("CUAP"), fr=2
    )

def make_lecap(ticker, emision, venc, tem, price=None):
    return lecaps(
        name=ticker,
        start_date=pd.to_datetime(emision, dayfirst=True).to_pydatetime(),
        end_date=pd.to_datetime(venc, dayfirst=True).to_pydatetime(),
        tem=float(tem)/100 if tem>1 else float(tem),  # accepts 3.95 -> 0.0395 or 0.0395
        price=price if price is not None else px_lookup(ticker)
    )

def make_dlk(ticker, emision, venc, price=None):
    mep = mep_al30 if mep_al30 is not None else 1.0
    return dlk(
        name=ticker,
        start_date=pd.to_datetime(emision, dayfirst=True).to_pydatetime(),
        end_date=pd.to_datetime(venc, dayfirst=True).to_pydatetime(),
        mep=float(mep),
        price=price if price is not None else px_lookup(ticker)
    )

# Sample rows (you can paste your full lists)
lecap_rows = [
    ("S12S5","12/9/2025","13/9/2024",3.95, "Fija"),
    ("S30S5","30/9/2025","30/9/2024",3.98, "Fija"),
    ("T17O5","17/10/2025","14/10/2024",3.90, "Fija"),
    ("S31O5","31/10/2025","16/12/2024",2.74, "Fija"),
    ("S10N5", "10/11/2025","31/01/2025", 2.2, "Fija"),
    ("S28N5","28/11/2025","14/2/2025",2.26, "Fija"),
    ("T15D5","15/12/2025","14/10/2024",3.89, "Fija"),
    ("S16E6","16/01/2026","18/08/2025",3.6, "Fija"),
    ("T30E6","30/1/2026","16/12/2024",2.65, "Fija"),
    ("T13F6","13/2/2026","29/11/2024",2.60, "Fija"),
    ("S27F6","29/2/2026","29/8/2025",3.95, "Fija"),
    ("S29Y6","29/5/2026","30/5/2025",2.35, "Fija"),
    ("T30J6","30/6/2026","17/1/2025",2.15, "Fija"),
    ("T15E7","15/1/2027","31/1/2025",2.05, "Fija"),
    ("TTM26","16/3/2026","29/1/2025", 2.225, "Fija"),
    ("TTJ26","30/6/2026","29/1/2025", 2.19, "Fija"),
    ("TTS26","15/9/2026","29/01/2025", 2.17, "Fija"),
    ("TTD26","15/12/2026","29/01/2025", 2.14, "Fija")
]
dlk_rows = [
    ("D31O5", "10/07/2025", "31/10/2025", "Dolar Linked"),
    ("TZVD5", "01/07/2024", "15/12/2025", "Dolar Linked"),
    ("D16E6", "28/04/2025", "16/01/2026", "Dolar Linked"),
    ("TZV26", "28/02/2024", "30/06/2026", "Dolar Linked")
]

# Instantiate defaults
cer_objs = [make_tx25(), make_tx26(), make_tx28(), make_dicp(), make_cuap()]
lecap_objs = [make_lecap(*r) for r in lecap_rows]
dlk_objs = [make_dlk(*r) for r in dlk_rows]

# -----------------------------
# METRICS TABLES (Plotly)
# -----------------------------
def summarize_bonds(objs, tipo):
    rows = []
    for b in objs:
        try:
            xirr = b.xirr()
            tna30 = b.tna30() if hasattr(b, "tna30") else np.nan
            dur = b.duration() if hasattr(b, "duration") else np.nan
            md  = b.modified_duration() if hasattr(b, "modified_duration") else np.nan
            conv = b.convexity() if hasattr(b, "convexity") else np.nan
            cy = b.current_yield() if hasattr(b, "current_yield") else np.nan
            pago = b.cash_flow()[-1]
        except Exception:
            xirr=tna30=dur=md=conv=cy=pago=np.nan
        rows.append({
            "Ticker": b.name,
            "Tipo": tipo,
            "Vencimiento": pd.to_datetime(getattr(b, "end_date", datetime.today())).strftime("%d/%m/%y"),
            "Precio": round(getattr(b, "price", np.nan), 2),
            "Pago": round(pago,2) if pd.notna(pago) else np.nan,
            "TIREA": round(xirr,2) if pd.notna(xirr) else np.nan,
            "TNA 30": round(tna30,2) if pd.notna(tna30) else np.nan,
            "Dur": round(dur,2) if pd.notna(dur) else np.nan,
            "Mod Dur": round(md,2) if pd.notna(md) else np.nan,
            "Conv": round(conv,2) if pd.notna(conv) else np.nan,
            "CY": round(cy,2) if pd.notna(cy) else np.nan
        })
    return pd.DataFrame(rows)

def df_to_plotly_table(df: pd.DataFrame, title=None):
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color='paleturquoise', align='center'),
        cells=dict(values=[df[c] for c in df.columns], fill_color='lavender', align='center')
    )])
    if title:
        fig.update_layout(title=title)
    return fig

st.title("🇦🇷 Bonos & Letras — Métricas")
k1, k2 = st.columns(2)
k1.metric("Filas precios (ocultas)", f"{len(df_all):,}")
k2.metric("MEP AL30 (Bid)", f"{mep_al30:,.2f}" if isinstance(mep_al30,(int,float)) else "—")

st.divider()

# Tables (NO raw df_all shown)
df_cer_tab = summarize_bonds(cer_objs, "CER")
df_lecap_tab = summarize_bonds(lecap_objs, "LECAP")
df_dlk_tab = summarize_bonds(dlk_objs, "DLK")

t1, t2, t3 = st.tabs(["CER", "LECAPs", "DLK"])
with t1:
    st.plotly_chart(df_to_plotly_table(df_cer_tab, "CER"), use_container_width=True)
with t2:
    st.plotly_chart(df_to_plotly_table(df_lecap_tab, "LECAPs"), use_container_width=True)
with t3:
    st.plotly_chart(df_to_plotly_table(df_dlk_tab, "DLK"), use_container_width=True)

st.divider()

# -----------------------------
# INTERACTIVE: override price & recompute fundamentals
# -----------------------------
st.subheader("🛠️ Recalcular métricas con precio manual")

tipo = st.selectbox("Tipo de instrumento", ["CER", "LECAP", "DLK"], index=0)

if tipo == "CER":
    names = [o.name for o in cer_objs]
    which = st.selectbox("Elegí el bono CER", names)
    obj = next(o for o in cer_objs if o.name == which)
    default_px = obj.price if obj.price==obj.price else 100.0
    new_px = st.number_input("Precio (clean) por 100 nominal", min_value=0.01, value=float(default_px), step=0.01)
    # Rebuild with new price
    maker = {"TX25": make_tx25, "TX26": make_tx26, "TX28": make_tx28, "DICP": make_dicp, "CUAP": make_cuap}[which]
    new_obj = maker(price=new_px)

elif tipo == "LECAP":
    names = [o.name for o in lecap_objs]
    which = st.selectbox("Elegí la letra", names)
    obj = next(o for o in lecap_objs if o.name == which)
    default_px = obj.price if obj.price==obj.price else 100.0
    new_px = st.number_input("Precio por 100 nominal", min_value=0.01, value=float(default_px), step=0.01)
    # Rebuild on the fly
    row = next(r for r in lecap_rows if r[0]==which)
    new_obj = make_lecap(*row, price=new_px)

else:  # DLK
    names = [o.name for o in dlk_objs]
    which = st.selectbox("Elegí el DLK", names)
    obj = next(o for o in dlk_objs if o.name == which)
    default_px = obj.price if obj.price==obj.price else 100.0
    new_px = st.number_input("Precio por 100 nominal", min_value=0.01, value=float(default_px), step=0.01)
    # Rebuild on the fly
    row = next(r for r in dlk_rows if r[0]==which)
    new_obj = make_dlk(*row, price=new_px)

# Compute fundamentals
def safe(fn):
    try:
        return fn()
    except Exception:
        return np.nan

colA, colB, colC, colD, colE, colF = st.columns(6)
xirr = safe(new_obj.xirr)
tna30 = safe(new_obj.tna30) if hasattr(new_obj,"tna30") else np.nan
dur = safe(new_obj.duration) if hasattr(new_obj,"duration") else np.nan
md  = safe(new_obj.modified_duration) if hasattr(new_obj,"modified_duration") else np.nan
cx  = safe(new_obj.convexity) if hasattr(new_obj,"convexity") else np.nan
cy  = safe(new_obj.current_yield) if hasattr(new_obj,"current_yield") else np.nan

colA.metric("TIREA (%)", "—" if pd.isna(xirr) else f"{xirr:.2f}")
colB.metric("TNA 30 (%)", "—" if pd.isna(tna30) else f"{tna30:.2f}")
colC.metric("Duración (años)", "—" if pd.isna(dur) else f"{dur:.2f}")
colD.metric("Mod. Dur", "—" if pd.isna(md) else f"{md:.2f}")
colE.metric("Convexidad", "—" if pd.isna(cx) else f"{cx:.2f}")
colF.metric("Current Yield (%)", "—" if pd.isna(cy) else f"{cy:.2f}")

st.caption("Las tablas superiores son Plotly Tables. El dataset de precios (`df_all`) no se muestra en pantalla.")

