# app.py
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ===========================
# Public RAW URL for your Excel (GitHub)
# ===========================
RAW_XLSX_URL = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"

# ===========================
# ons_pro class (stub-aware) + pure-Python IRR (no SciPy)
# ===========================
class ons_pro:
    def __init__(self, name, empresa, curr, law, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price, fr):
        self.name = name
        self.empresa = empresa
        self.curr = curr
        self.law = law
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = payment_frequency   # months between coupons
        self.amortization_dates = amortization_dates # ["YYYY-MM-DD", ...]
        self.amortizations = amortizations           # amounts per 100 nominal
        self.rate = rate / 100.0                     # store as decimal p.a.
        self.price = price                           # clean price per 100 nominal
        self.frequency = fr                          # coupons per year (e.g., 2 = semi)

    def _as_dt(self, d):
        return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")

    def outstanding_on(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        ref_date = self._as_dt(ref_date)
        paid = sum(a for d, a in zip(self.amortization_dates, self.amortizations)
                   if self._as_dt(d) <= ref_date)
        return max(0.0, 100.0 - paid)

    def generate_payment_dates(self):
        """
        Returns list of 'YYYY-MM-DD': [t_settlement, D1, D2, ..., end_date]
        Built backwards from end_date in steps of payment_frequency (months).
        Respects initial stub. Only future coupon dates (> settlement) are kept.
        """
        settlement = datetime.today() + timedelta(days=1)
        back = []
        cur = self.end_date if isinstance(self.end_date, datetime) else datetime.strptime(self.end_date, "%Y-%m-%d")
        start = self.start_date if isinstance(self.start_date, datetime) else datetime.strptime(self.start_date, "%Y-%m-%d")

        back.append(cur)  # maturity
        while True:
            prev = cur - relativedelta(months=self.payment_frequency)
            if prev <= start:
                break
            back.append(prev)
            cur = prev

        schedule = sorted(back)
        future_schedule = [d for d in schedule if d > settlement]
        return [settlement.strftime("%Y-%m-%d")] + [d.strftime("%Y-%m-%d") for d in future_schedule]

    def residual_value(self):
        """Residual (per 100) en cada fecha de la grilla, contemplando amortizaciones previas."""
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        return [self.outstanding_on(d) for d in dates]

    def accrued_interest(self, ref_date=None):
        """
        Interés corrido ACT/365F con base del INICIO de período (mismo criterio que coupon_payments).
        """
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        ref_date = self._as_dt(ref_date)
    
        dates_str = self.generate_payment_dates()
        all_dt = [datetime.strptime(s, "%Y-%m-%d") for s in dates_str]
        coup_dt = all_dt[1:]  # fechas de cupón
    
        if not coup_dt:
            return 0.0
    
        next_coupon = None
        for d in coup_dt:
            if d > ref_date:
                next_coupon = d
                break
        if next_coupon is None:
            return 0.0
    
        idx = coup_dt.index(next_coupon)
        if idx == 0:
            period_start = max(self._as_dt(self.start_date), next_coupon - relativedelta(months=self.payment_frequency))
        else:
            period_start = coup_dt[idx - 1]
    
        residual_at_start = self.outstanding_on(period_start)
        full_coupon = (self.rate / self.frequency) * residual_at_start
    
        total_days = max(1, (next_coupon - period_start).days)
        accrued_days = max(0, min((ref_date - period_start).days, total_days))
        return full_coupon * (accrued_days / total_days)

    def parity(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        vt = self.outstanding_on(ref_date) + self.accrued_interest(ref_date)
        return float('nan') if vt == 0 else round(self.price / vt * 100, 2)

    def amortization_payments(self):
        cap, am = [], dict(zip(self.amortization_dates, self.amortizations))
        for d in self.generate_payment_dates():
            cap.append(am.get(d, 0.0))
        return cap

    def coupon_payments(self):
        """
        Cupón del período = (tasa anual / frecuencia) * residual al INICIO del período.
        INICIO = cupón anterior; si no existe (primer futuro), uso max(start_date, next_coupon - payment_frequency meses).
        """
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        coupons = [0.0]  # t0 = settlement
        coupon_dates = dates[1:]  # solo fechas de cupón
    
        for i, cdate in enumerate(coupon_dates):
            if i == 0:
                period_start = max(self._as_dt(self.start_date), cdate - relativedelta(months=self.payment_frequency))
            else:
                period_start = coupon_dates[i - 1]
    
            residual_at_start = self.outstanding_on(period_start)
            coupons.append((self.rate / self.frequency) * residual_at_start)
    
        return coupons

    def cash_flow(self):
        cfs, caps, cpns = [], self.amortization_payments(), self.coupon_payments()
        for i, _ in enumerate(self.generate_payment_dates()):
            cfs.append(-self.price if i == 0 else caps[i] + cpns[i])
        return cfs

    def xnpv(self, _dates=None, _cash_flow=None, rate_custom=0.08):
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        d0 = datetime.today() + timedelta(days=1)
        return sum(cf / (1.0 + rate_custom) ** ((dt - d0).days / 365.0)
                   for cf, dt in zip(self.cash_flow(), dates))

    def xirr(self):
        """Pure-Python XIRR via auto-bracket + bisection; returns % E.A."""
        def f(r): return self.xnpv(rate_custom=r)
        lows  = [-0.99, -0.50, -0.20, -0.10, 0.0]
        highs = [0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
        for lo in lows:
            flo = f(lo)
            if np.isnan(flo): continue
            for hi in highs:
                if hi <= lo: continue
                fhi = f(hi)
                if np.isnan(fhi): continue
                if flo == 0:  return round(lo * 100.0, 2)
                if fhi == 0:  return round(hi * 100.0, 2)
                if flo * fhi < 0:
                    a, b = lo, hi
                    fa, fb = flo, fhi
                    for _ in range(100):
                        m = 0.5 * (a + b); fm = f(m)
                        if abs(fm) < 1e-12 or (b - a) < 1e-10:
                            return round(m * 100.0, 2)
                        if fa * fm <= 0: b, fb = m, fm
                        else: a, fa = m, fm
                    return round(0.5 * (a + b) * 100.0, 2)
        return float('nan')

    def tna_180(self):
        irr = self.xirr() / 100.0
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 2)

    def duration(self):
        irr = self.xirr() / 100.0
        d0  = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        flows = [(cf, dt) for i, (cf, dt) in enumerate(zip(self.cash_flow(), dates)) if i > 0]
        pv_price = sum(cf / (1 + irr) ** ((dt - d0).days / 365.0) for cf, dt in flows)
        mac = sum(((dt - d0).days / 365.0) * (cf / (1 + irr) ** ((dt - d0).days / 365.0))
                  for cf, dt in flows) / (pv_price if pv_price != 0 else np.nan)
        return round(mac, 2)

    def modified_duration(self):
        irr = self.xirr() / 100.0
        dur = self.duration()
        return round(dur / (1 + irr), 2)

    def convexity(self):
        y = self.xirr() / 100.0
        d0 = datetime.today() + timedelta(days=1)
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        flows = [(cf, (dt - d0).days / 365.0)
                 for i, (cf, dt) in enumerate(zip(self.cash_flow(), dates)) if i > 0]
        pv = sum(cf / (1 + y) ** t for cf, t in flows)
        if pv == 0: return float('nan')
        cx = sum(cf * t * (t + 1) / (1 + y) ** (t + 2) for cf, t in flows) / pv
        return round(cx, 2)

    def current_yield(self):
        cpns = self.coupon_payments()
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        future_idx = [i for i, d in enumerate(dates)
                      if d > (datetime.today() + timedelta(days=1)) and cpns[i] > 0]
        if not future_idx: return float('nan')
        i0 = future_idx[0]
        n = min(self.frequency, len(cpns) - i0)
        annual_coupons = sum(cpns[i0:i0 + n])
        return round(annual_coupons / self.price * 100.0, 2)

# ===========================
# Helpers: parsing / normalization / loading
# ===========================
def parse_date_cell(s):
    if pd.isna(s): return None
    if isinstance(s, datetime): return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try: return datetime.strptime(s, fmt)
        except ValueError: pass
    return pd.to_datetime(s, dayfirst=True, errors="raise").to_pydatetime()

def parse_date_list(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    parts = str(cell).replace(",", "/").split(";")
    return [parse_date_cell(p).strftime("%Y-%m-%d") for p in parts]

def parse_float_cell(x):
    if pd.isna(x): return np.nan
    s = str(x).strip().replace("%", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")  # "1.234,56" -> "1234.56"
    else:
        s = s.replace(",", ".")
    try: return float(s)
    except Exception: return np.nan

def normalize_rate_to_percent(rate_raw):
    if pd.isna(rate_raw): return np.nan
    r = float(rate_raw)
    return r * 100.0 if r < 1 else r

def parse_amorts(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    return [parse_float_cell(p) for p in str(cell).split(";")]

def get_price_for_symbol(df_all, symbol, prefer="px_bid"):
    row = df_all.loc[df_all["symbol"] == symbol]
    if row.empty: return np.nan
    prefer_col = prefer if prefer in row.columns else None
    alt_col = "px_ask" if prefer == "px_bid" else "px_bid"
    alt_col = alt_col if alt_col in row.columns else None
    if prefer_col and not pd.isna(row.iloc[0][prefer_col]): return float(row.iloc[0][prefer_col])
    if alt_col and not pd.isna(row.iloc[0][alt_col]):       return float(row.iloc[0][alt_col])
    for c in row.columns:
        if c != "symbol":
            try: return float(row.iloc[0][c])
            except Exception: continue
    return np.nan

def load_ons_from_excel(path_or_buffer, df_all, price_col_prefer="px_bid"):
    """
    Required columns: name, empresa, curr, law, start_date, end_date,
                      payment_frequency, amortization_dates, amortizations, rate, fr
    """
    raw = pd.read_excel(path_or_buffer, dtype=str)

    required = ["name","empresa","curr","law","start_date","end_date",
                "payment_frequency","amortization_dates","amortizations","rate","fr"]
    miss = [c for c in required if c not in raw.columns]
    if miss: raise ValueError(f"Missing columns in Excel: {miss}")

    bonds = []
    for _, r in raw.iterrows():
        name  = str(r["name"]).strip()
        emp   = str(r["empresa"]).strip()
        curr  = str(r["curr"]).strip()
        law   = str(r["law"]).strip()
        start = parse_date_cell(r["start_date"])
        end   = parse_date_cell(r["end_date"])
        pay_freq = int(parse_float_cell(r["payment_frequency"]))

        am_dates = parse_date_list(r["amortization_dates"])
        am_amts  = parse_amorts(r["amortizations"])
        if len(am_dates) != len(am_amts):
            if len(am_dates) == 1 and len(am_amts) == 0:
                am_amts = [100.0]
            elif len(am_dates) == 0 and len(am_amts) == 1:
                am_dates = [end.strftime("%Y-%m-%d")]
            else:
                raise ValueError(f"{name}: amortization dates/amounts mismatch ({am_dates} vs {am_amts})")

        rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))
        fr = int(parse_float_cell(r["fr"]))
        price = get_price_for_symbol(df_all, name, prefer=price_col_prefer)

        bonds.append(ons_pro(
            name=name, empresa=emp, curr=curr, law=law,
            start_date=start, end_date=end,
            payment_frequency=pay_freq,
            amortization_dates=am_dates, amortizations=am_amts,
            rate=rate_pct, price=price, fr=fr
        ))
    return bonds

def bond_fundamentals_ons(bond_objs):
    rows = []
    for b in bond_objs:
        try:
            rows.append([
                b.name, b.empresa, b.curr, b.law, b.rate * 100, b.price,
                b.xirr(), b.tna_180(), b.duration(), b.modified_duration(),
                b.convexity(), b.current_yield(), b.parity()
            ])
        except Exception:
            rows.append([b.name, b.empresa, b.curr, b.law, b.rate * 100, b.price,
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    cols = ["Ticker","Empresa","Moneda de Pago","Ley","Cupón","Precio",
            "Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]
    df = pd.DataFrame(rows, columns=cols)

    for c in ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Cupón"] = df["Cupón"].round(4)
    df["Precio"] = df["Precio"].round(2)
    for c in ["Yield","TNA_180","Current Yield","Paridad (%)"]:
        df[c] = df[c].round(2)
    for c in ["Dur","MD","Conv"]:
        df[c] = df[c].round(2)
    return df

# ===========================
# Data sources: df_all (prices) and Excel (specs)
# ===========================
@st.cache_data(show_spinner=False)
def fetch_df_all_from_endpoints():
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

    df_bonds = to_df(fetch_json(url_bonds)); df_bonds["source"] = "bonds"
    df_notes = to_df(fetch_json(url_notes)); df_notes["source"] = "notes"
    df_corps = to_df(fetch_json(url_corps)); df_corps["source"] = "corps"

    df_all = pd.concat([df_bonds, df_notes, df_corps], ignore_index=True, sort=False)

    # normalize column names
    if "symbol" not in df_all.columns and "ticker" in df_all.columns:
        df_all = df_all.rename(columns={"ticker": "symbol"})
    guess = {"bid":"px_bid","ask":"px_ask","px_bid_":"px_bid","px_ask_":"px_ask",
             "price_bid":"px_bid","price_ask":"px_ask"}
    for old, new in guess.items():
        if old in df_all.columns and new not in df_all.columns:
            df_all = df_all.rename(columns={old: new})
    return df_all

@st.cache_data(show_spinner=False)
def get_excel_from_repo():
    """Use local file if present; otherwise fetch from GitHub Raw."""
    local = Path(__file__).parent / "listado_ons.xlsx"
    if local.exists():
        return str(local)  # path works for pd.read_excel
    r = requests.get(RAW_XLSX_URL, timeout=25)
    r.raise_for_status()
    return io.BytesIO(r.content)       # file-like also works

@st.cache_data(show_spinner=False)
def load_everything(excel_source, df_all, prefer_col):
    bonds = load_ons_from_excel(excel_source, df_all, price_col_prefer=prefer_col)
    return bond_fundamentals_ons(bonds)

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="ONs — Fundamentals", page_icon="💼", layout="wide")
st.title("💼 ONs — Fundamentals")

with st.sidebar:
    st.header("⚙️ Settings")
    prefer_col = st.selectbox("Preferred price", options=["px_bid", "px_ask"], index=0)
    reload_btn = st.button("🔄 Reload data")

# Build df_all from endpoints
with st.spinner("Fetching prices (data912)…"):
    try:
        df_all = fetch_df_all_from_endpoints()
    except Exception as e:
        st.error(f"❌ Could not fetch prices: {e}")
        st.stop()

# Excel source (from repo / GitHub Raw)
try:
    excel_source = get_excel_from_repo()
except Exception as e:
    st.error(f"❌ Could not load Excel: {e}")
    st.stop()

# Allow manual reload
if reload_btn:
    fetch_df_all_from_endpoints.clear()
    get_excel_from_repo.clear()
    load_everything.clear()
    df_all = fetch_df_all_from_endpoints()
    excel_source = get_excel_from_repo()

# Compute metrics
with st.spinner("Calculating metrics…"):
    try:
        df_metrics = load_everything(excel_source, df_all, prefer_col)
    except Exception as e:
        st.error(f"❌ Error computing metrics: {e}")
        st.stop()

# Filters
st.subheader("Filters")
empresas = sorted([e for e in df_metrics["Empresa"].dropna().unique()])
sel_empresas = st.multiselect("Empresa", empresas, default=empresas)
df_view = df_metrics[df_metrics["Empresa"].isin(sel_empresas)].reset_index(drop=True)

# Table
st.subheader("Metrics table")
col_config = {
    "Cupón": st.column_config.NumberColumn("Cupón (%)", format="%.4f"),
    "Precio": st.column_config.NumberColumn("Precio", format="%.2f"),
    "Yield": st.column_config.NumberColumn("Yield (%)", format="%.2f"),
    "TNA_180": st.column_config.NumberColumn("TNA 180 (%)", format="%.2f"),
    "Dur": st.column_config.NumberColumn("Duración (años)", format="%.2f"),
    "MD": st.column_config.NumberColumn("Mod. Duration (años)", format="%.2f"),
    "Conv": st.column_config.NumberColumn("Convexidad", format="%.2f"),
    "Current Yield": st.column_config.NumberColumn("Current Yield (%)", format="%.2f"),
    "Paridad (%)": st.column_config.NumberColumn("Paridad (%)", format="%.2f"),
}
st.dataframe(df_view, use_container_width=True, hide_index=True, column_config=col_config)

# Downloads
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download CSV (filtered)",
        data=df_view.to_csv(index=False).encode("utf-8-sig"),
        file_name="ons_fundamentals.csv",
        mime="text/csv"
    )
with c2:
    try:
        import xlsxwriter  # noqa: F401
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            df_view.to_excel(writer, index=False, sheet_name="Fundamentals")
        st.download_button(
            "⬇️ Download Excel (filtered)",
            data=buffer.getvalue(),
            file_name="ons_fundamentals.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.caption("Install `xlsxwriter` to enable Excel export.")
