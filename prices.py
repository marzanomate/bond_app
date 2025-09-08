# app.py — Calculadora de ONs + Soberanos (versión optimizada y comentada)

import re  # Utilidades de expresiones regulares
import io  # BytesIO para manejar contenido binario en memoria
import numpy as np  # Cálculo numérico y vectorización
import pandas as pd  # Manipulación de datos tabulares
import requests  # Peticiones HTTP para descargar JSON/Excel
import streamlit as st  # Framework de la app
from datetime import datetime, timedelta  # Fechas básicas
from dateutil.relativedelta import relativedelta  # Desplazamientos de meses exactos
from concurrent.futures import ThreadPoolExecutor, as_completed  # Concurrencia para IO

# =====================================
# Config
# =====================================
st.set_page_config(page_title="Calculadora ONs + Soberanos", layout="wide")  # Título y layout ancho

EXCEL_URL_DEFAULT = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"  # Excel remoto por defecto
URL_BONDS = "https://data912.com/live/arg_bonds"   # Endpoint precios bonos soberanos/ONs
URL_NOTES = "https://data912.com/live/arg_notes"   # Endpoint notas/otros
URL_CORPS = "https://data912.com/live/arg_corp"    # Endpoint corporativos

# =====================================
# Helpers cacheados de red
# =====================================

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_excel_bytes(url: str) -> bytes:
    """Descarga el Excel (bytes). Cacheado 1h para evitar red en cada rerun."""
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.content

@st.cache_data(ttl=90, show_spinner=False)
def fetch_json(url: str):
    """Descarga JSON (cacheado 90s)."""
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

# =====================================
# Normalización de respuestas de precios
# =====================================

def to_df(payload):
    """Convierte payload (dict/list) a DataFrame plano."""
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    return pd.json_normalize(payload)

def harmonize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica columnas a ['symbol','px_bid','px_ask', ...]."""
    rename_map = {}
    cols = {c.lower(): c for c in df.columns}
    if "ticker" in cols and "symbol" not in df.columns:
        rename_map[cols["ticker"]] = "symbol"
    if "bid" in cols and "px_bid" not in df.columns:
        rename_map[cols["bid"]] = "px_bid"
    if "ask" in cols and "px_ask" not in df.columns:
        rename_map[cols["ask"]] = "px_ask"
    out = df.rename(columns=rename_map)
    for c in ["symbol", "px_bid", "px_ask"]:
        if c not in out.columns:
            out[c] = np.nan
    return out[["symbol", "px_bid", "px_ask"] + [c for c in out.columns if c not in ["symbol","px_bid","px_ask"]]]

@st.cache_data(ttl=90, show_spinner=False)
def build_df_all() -> pd.DataFrame:
    """Descarga 3 endpoints en paralelo (IO-bound) y concatena precios."""
    urls = [URL_BONDS, URL_NOTES, URL_CORPS]
    frames = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(fetch_json, u): u for u in urls}
        for fut in as_completed(futs):
            try:
                df = to_df(fut.result())
                if not df.empty:
                    frames.append(harmonize_prices(df))
            except Exception:
                pass
    if not frames:
        return pd.DataFrame(columns=["symbol","px_bid","px_ask"])
    df_all = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["symbol"])
    for c in ["px_bid","px_ask"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    return df_all

def get_price_for_symbol(df_all: pd.DataFrame, symbol: str, prefer: str = "px_ask") -> float:
    """Busca precio preferido por símbolo; si no hay, prueba la alternativa."""
    row = df_all.loc[df_all["symbol"] == symbol]
    if row.empty:
        raise KeyError(f"No encontré {symbol} en df_all['symbol']")
    if prefer in row.columns and pd.notna(row.iloc[0][prefer]):
        return float(row.iloc[0][prefer])
    alt = "px_bid" if prefer == "px_ask" else "px_ask"
    if alt in row.columns and pd.notna(row.iloc[0][alt]):
        return float(row.iloc[0][alt])
    raise KeyError(f"{symbol}: no hay {prefer} ni {alt} con precio válido")

# =====================================
# Parseo de celdas del Excel
# =====================================

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def parse_date_cell(s):
    """Convierte múltiples formatos de fecha en datetime."""
    if pd.isna(s):
        return None
    if isinstance(s, (datetime, pd.Timestamp)):
        return pd.Timestamp(s).to_pydatetime()
    s = str(s).strip().replace("\u00A0", " ")
    token = s.split("T")[0].split()[0]
    if ISO_DATE_RE.match(token):
        return datetime.strptime(token, "%Y-%m-%d")
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(token, fmt)
        except ValueError:
            pass
    return pd.to_datetime(token, dayfirst=True, errors="raise").to_pydatetime()

def parse_date_list(cell):
    """Split por ';' y parsea fechas a strings YYYY-MM-DD."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    parts = str(cell).replace(",", "/").split(";")
    out = []
    for p in parts:
        d = parse_date_cell(p)
        out.append(d.strftime("%Y-%m-%d"))
    return out

def parse_float_cell(x):
    """Convierte strings con comas/puntos/porcentajes a float."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def normalize_rate_to_percent(r):
    """Si <1 asume decimal (0.12 -> 12%); si >=1 asume ya en % (12 -> 12%)."""
    if pd.isna(r):
        return np.nan
    r = float(r)
    return r * 100.0 if r < 1 else r

def parse_amorts(cell):
    """Parsea lista de amortizaciones separadas por ';' a floats."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    return [parse_float_cell(p) for p in str(cell).split(";")]

# =====================================
# Clase de ON (con memoización y XIRR vectorizado)
# =====================================

class ons_pro:
    """Modelo de ON con generación de flujos y métricas."""
    def __init__(self, name, empresa, curr, law, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price):
        self.name = name
        self.empresa = empresa
        self.curr = curr
        self.law = law
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = int(payment_frequency)
        if self.payment_frequency <= 0:
            raise ValueError(f"{name}: payment_frequency debe ser > 0")
        self.amortization_dates = amortization_dates
        self.amortizations = amortizations
        self.rate = float(rate) / 100.0
        self.price = float(price)
        self._memo = {}

    def _freq(self):
        return max(1, int(round(12 / self.payment_frequency)))

    def _as_dt(self, d):
        return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")

    def outstanding_on(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        ref_date = self._as_dt(ref_date)
        paid = sum(a for d, a in zip(self.amortization_dates, self.amortizations)
                   if self._as_dt(d) <= ref_date)
        return max(0.0, 100.0 - paid)

    def _schedule(self):
        if "schedule" in self._memo:
            return self._memo["schedule"]
        settlement = datetime.today() + timedelta(days=1)
        back = []
        cur = self._as_dt(self.end_date)
        start = self._as_dt(self.start_date)
        back.append(cur)
        while True:
            prev = cur - relativedelta(months=self.payment_frequency)
            if prev <= start:
                break
            back.append(prev)
            cur = prev
        schedule = [settlement] + sorted([d for d in back if d > settlement])
        self._memo["schedule"] = schedule
        return schedule

    def generate_payment_dates(self):
        return [d.strftime("%Y-%m-%d") for d in self._schedule()]

    def amortization_payments(self):
        cap = []
        dates = self.generate_payment_dates()
        am = dict(zip(self.amortization_dates, self.amortizations))
        for d in dates:
            cap.append(am.get(d, 0.0))
        return cap

    def coupon_payments(self):
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        coupons = [0.0]
        coupon_dates = dates[1:]
        f = self._freq()
        for i, cdate in enumerate(coupon_dates):
            period_start = (max(self._as_dt(self.start_date),
                                cdate - relativedelta(months=self.payment_frequency))
                            if i == 0 else coupon_dates[i-1])
            base = self.outstanding_on(period_start)
            coupons.append((self.rate / f) * base)
        return coupons

    def residual_value(self):
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        return [self.outstanding_on(d) for d in dates]

    def cash_flow(self):
        cfs = []
        dates = self.generate_payment_dates()
        caps = self.amortization_payments()
        cpns = self.coupon_payments()
        for i, _ in enumerate(dates):
            cfs.append(-self.price if i == 0 else caps[i] + cpns[i])
        return cfs

    def _times_and_flows(self):
        if "tf" in self._memo:
            return self._memo["tf"]
        d0 = datetime.today() + timedelta(days=1)
        dates = self._schedule()
        caps = self.amortization_payments()
        cpns = self.coupon_payments()
        cfs = [-self.price] + [c + a for c, a in zip(cpns[1:], caps[1:])]
        t_years = np.array([(dt - d0).days / 365.0 for dt in dates], dtype=float)
        cfs_arr = np.array(cfs, dtype=float)
        self._memo["tf"] = (t_years, cfs_arr)
        return self._memo["tf"]

    def xnpv_vec(self, r):
        t, c = self._times_and_flows()
        return float(np.sum(c / (1.0 + r) ** t))

    def dxnpv_vec(self, r):
        t, c = self._times_and_flows()
        return float(np.sum(-t * c / (1.0 + r) ** (t + 1)))

    def xirr(self):
        guess = 0.25
        r = guess
        for _ in range(12):
            f = self.xnpv_vec(r)
            df = self.dxnpv_vec(r)
            if not np.isfinite(f) or not np.isfinite(df) or df == 0:
                break
            r_new = r - f / df
            if r_new <= -0.999:
                r_new = (r - 0.999) / 2
            if abs(r_new - r) < 1e-10:
                return round(r_new * 100.0, 2)
            r = r_new
        lo, hi = -0.9, 5.0
        flo, fhi = self.xnpv_vec(lo), self.xnpv_vec(hi)
        if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:
            return float('nan')
        for _ in range(40):
            m = 0.5 * (lo + hi)
            fm = self.xnpv_vec(m)
            if abs(fm) < 1e-12:
                return round(m * 100.0, 2)
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return round(0.5 * (lo + hi) * 100.0, 2)

    def tna_180(self):
        irr = self.xirr() / 100.0
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 2)

    def duration(self):
        irr = self.xirr() / 100.0
        d0 = datetime.today() + timedelta(days=1)
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        cfs = self.cash_flow()
        flows = [(cf, dt) for i, (cf, dt) in enumerate(zip(cfs, dates)) if i > 0 and cf != 0]
        if not flows:
            return float('nan')
        pv_price = sum(cf / (1 + irr) ** ((dt - d0).days / 365.0) for cf, dt in flows)
        if pv_price == 0 or np.isnan(pv_price):
            return float('nan')
        mac = sum(((dt - d0).days / 365.0) * (cf / (1 + irr) ** ((dt - d0).days / 365.0))
                  for cf, dt in flows) / pv_price
        return round(mac, 2)

    def modified_duration(self):
        irr = self.xirr() / 100.0
        dur = self.duration()
        den = 1 + irr
        if den == 0 or np.isnan(den) or np.isnan(dur):
            return float('nan')
        return round(dur / den, 2)

    def convexity(self):
        y = self.xirr() / 100.0
        d0 = datetime.today() + timedelta(days=1)
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        cfs = self.cash_flow()
        flows = [(cf, (dt - d0).days / 365.0) for i, (cf, dt) in enumerate(zip(cfs, dates)) if i > 0 and cf != 0]
        if not flows:
            return float('nan')
        pv = sum(cf / (1 + y) ** t for cf, t in flows)
        if pv <= 0 or np.isnan(pv):
            return float('nan')
        cx = sum(cf * t * (t + 1) / (1 + y) ** (t + 2) for cf, t in flows) / pv
        return round(cx, 2)

    def current_yield(self):
        cpns = self.coupon_payments()
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        future_idx = [i for i, d in enumerate(dates)
                      if d > (datetime.today() + timedelta(days=1)) and cpns[i] > 0]
        if not future_idx:
            return float('nan')
        i0 = future_idx[0]
        n = min(self._freq(), len(cpns) - i0)
        annual_coupons = sum(cpns[i0:i0 + n])
        return round(annual_coupons / self.price * 100.0, 2)

    def parity(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        vt = self.outstanding_on(ref_date) + self.coupon_payments()[0]
        return float('nan') if vt == 0 else round(self.price / vt * 100.0, 2)

# =====================================
# Bonos soberanos / ARGENTs (GD/AL/AE/BP*)
# =====================================

class bond_calculator_pro:
    """
    Calculadora para bonos soberanos (GD/AL/AE/BPB/BPC/BPD).
    - rate y step_up en DECIMAL (0.05 = 5%).
    - fr: cupones por año (2 = semestral).
    - payment_frequency: meses entre pagos (6 = semestral).
    - amortizations: % sobre 100 nominal (suman 100 si amortiza completo).
    """
    def __init__(self, name, start_date, end_date, payment_frequency, amortization_dates, amortizations, rate, 
                 price, fr, step_up_dates, step_up, outstanding,
                 emisor="Tesoro Nacional", curr="USD", law="-"):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = int(payment_frequency)
        self.amortization_dates = [str(d) for d in amortization_dates]
        self.amortizations = [float(a) for a in amortizations]
        self.rate = float(rate) / 100.0 if rate >= 1 else float(rate)
        self.price = float(price)
        self.frequency = int(fr)
        self.step_up_dates = [str(d) for d in step_up_dates]
        self.step_up = [float(x) if x < 1 else float(x)/100.0 for x in step_up]
        self.outstanding = float(outstanding)
        self.emisor = emisor
        self.curr = curr
        self.law = law

    def generate_payment_dates(self):
        dates = []
        current_date = self.start_date
        settlement = datetime.today() + timedelta(days=1)
        dates.append(settlement.strftime("%Y-%m-%d"))
        while current_date <= self.end_date:
            if current_date > settlement:
                dates.append(current_date.strftime("%Y-%m-%d"))
            current_date = current_date + relativedelta(months=self.payment_frequency)
        return dates
        
    def residual_value(self):
        residual = []
        current_residual = 100.0
        dates = self.generate_payment_dates()
        am_map = {d: float(a) for d, a in zip(self.amortization_dates, self.amortizations)}
        for date in dates:
            if date in am_map:
                current_residual = max(0.0, current_residual - am_map[date])
            residual.append(current_residual)
        return residual
        
    def amortization_payments(self):
        capital_payments = []
        dates = self.generate_payment_dates()
        am_map = {d: float(a) for d, a in zip(self.amortization_dates, self.amortizations)}
        for d in dates:
            capital_payments.append(am_map.get(d, 0.0))
        return capital_payments

    def step_up_rate(self):
        dates = self.generate_payment_dates()
        if not self.step_up_dates:
            return [self.rate] * len(dates)
        step = list(zip(self.step_up_dates, self.step_up))
        step.sort(key=lambda x: x[0])
        out = []
        for s in dates:
            sdt = datetime.strptime(s, "%Y-%m-%d")
            r = self.rate
            for d_str, r_step in step:
                d0 = datetime.strptime(d_str, "%Y-%m-%d")
                if sdt >= d0:
                    r = r_step
                else:
                    break
            out.append(r)
        return out

    def coupon_payments(self):
        dates = self.generate_payment_dates()
        rate_schedule = self.step_up_rate()
        residuals = self.residual_value()
        coupons = [0.0]
        for i in range(1, len(dates)):
            coupons.append((rate_schedule[i] / self.frequency) * residuals[i-1])
        return coupons

    def cash_flow(self):
        dates = self.generate_payment_dates()
        capital_payments = self.amortization_payments()
        coupon = self.coupon_payments()
        cash_flow = []
        for i in range(len(dates)):
            if i == 0:
                cash_flow.append(-self.price)
            else:
                cash_flow.append(capital_payments[i] + coupon[i])
        return cash_flow

    def xnpv(self, rate_custom = 0.08): 
        dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in self.generate_payment_dates()] 
        cash_flow = self.cash_flow()
        d0 = datetime.today() + timedelta(days=1)
        npv = sum([cf/(1.0 + rate_custom)**((date - d0).days/365.0) for cf, date in zip(cash_flow, dates)]) 
        return float(npv)

    def xirr(self):
        lo, hi = -0.95, 5.0
        flo, fhi = self.xnpv(lo), self.xnpv(hi)
        if np.isnan(flo) or np.isnan(fhi):
            return float('nan')
        if flo * fhi > 0:
            for hi_try in [10.0, 20.0]:
                fhi = self.xnpv(hi_try)
                if flo * fhi <= 0:
                    hi = hi_try
                    break
            else:
                return float('nan')
        for _ in range(60):
            m = 0.5 * (lo + hi)
            fm = self.xnpv(m)
            if abs(fm) < 1e-12:
                return round(m * 100.0, 2)
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return round(0.5 * (lo + hi) * 100.0, 2)

    def tna_180(self):
        irr = self.xirr()/100.0
        return round((((1+irr)**0.5 - 1)*2)*100.0, 2)

    def duration(self):
        irr = self.xirr()/100.0
        if not np.isfinite(irr):
            return float('nan')
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        cash_flow = self.cash_flow()
        d0 = datetime.today() + timedelta(days=1)
        times = np.array([(dt - d0).days/365.0 for dt in dates], dtype=float)
        disc = (1 + irr/self.frequency)**(self.frequency*times)
        pv_flows = np.array(cash_flow)/disc
        pv_flows[0] = 0.0
        pv = pv_flows.sum()
        if pv <= 0 or np.isnan(pv):
            return float('nan')
        mac = float((times * pv_flows).sum() / pv)
        return round(mac, 2)

    def modified_duration(self):
        dur = self.duration()
        irr = self.xirr()/100.0
        if not np.isfinite(dur) or not np.isfinite(irr):
            return float('nan')
        return round(dur / (1 + irr/self.frequency), 2)

    def convexity(self):
        irr = self.xirr()/100.0
        if not np.isfinite(irr):
            return float('nan')
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        cash_flow = self.cash_flow()
        d0 = datetime.today() + timedelta(days=1)
        times = np.array([(dt - d0).days/365.0 for dt in dates], dtype=float)
        disc = (1 + irr/self.frequency)**(self.frequency*times)
        pv = np.array(cash_flow)/disc
        pv[0] = 0.0
        price_pos = pv.sum()
        if price_pos <= 0 or np.isnan(price_pos):
            return float('nan')
        cx_term = (np.array(cash_flow) * times * (times + 1/self.frequency)) / \
                  ((1 + irr/self.frequency)**(self.frequency*times + 2))
        cx = float(cx_term[1:].sum() / price_pos)
        return round(cx, 2)

    def current_yield(self):
        cpns = self.coupon_payments()
        dates = [datetime.strptime(s, '%Y-%m-%d') for s in self.generate_payment_dates()]
        future_idx = [i for i, d in enumerate(dates) if d > (datetime.today() + timedelta(days=1)) and cpns[i] > 0]
        if not future_idx:
            return float('nan')
        i0 = future_idx[0]
        n = min(self.frequency, len(cpns) - i0)
        annual_coupons = sum(cpns[i0:i0 + n])
        return round(annual_coupons / self.price * 100.0, 2)

    def parity(self):
        vt = float(self.residual_value()[0])
        return float('nan') if vt == 0 else round(self.price / vt * 100.0, 2)

# --------------------------
# Helpers para integrar ambos tipos de objetos (ons_pro y soberanos)
# --------------------------

def _obj_metrics_row(b):
    """Devuelve una fila de métricas compatible con la grilla principal."""
    try:
        coupon_pct = float(b.rate) * 100.0
    except Exception:
        coupon_pct = float('nan')

    try:
        y = b.xirr()
        tna = b.tna_180()
        dur = b.duration()
        md = b.modified_duration()
        cx = b.convexity()
        cy = b.current_yield()
        par = b.parity() if hasattr(b, "parity") else float('nan')
    except Exception:
        y = tna = dur = md = cx = cy = par = float('nan')

    emisor = getattr(b, "emisor", getattr(b, "empresa", "-"))
    curr = getattr(b, "curr", getattr(b, "Moneda", "-"))
    ley = getattr(b, "law", getattr(b, "Ley", "-"))
    price = float(getattr(b, "price", float('nan')))

    return [
        getattr(b, "name", "-"), emisor, curr, ley,
        coupon_pct, price, y, tna, dur, md, cx, cy, par
    ]

def _metrics_columns():
    return [
        "Ticker","Emisor","Moneda de Pago","Ley","Cupón","Precio",
        "Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"
    ]

def bond_flows_frame(b) -> pd.DataFrame:
    """Tabla de flujos y saldos por fechas para un bono (ONS o soberano)."""
    dates = b.generate_payment_dates()
    try:
        res   = b.residual_value()
    except Exception:
        res = [np.nan]*len(dates)
    caps  = b.amortization_payments()
    cpns  = b.coupon_payments()
    cfs   = b.cash_flow()
    return pd.DataFrame({
        "Fecha": dates,
        "Residual": res,
        "Amortización": caps,
        "Cupón": cpns,
        "Flujo": cfs
    })

# =====================================
# Carga de ONs desde Excel
# =====================================

@st.cache_data(show_spinner=False)
def load_ons_from_excel(path_or_bytes, df_all: pd.DataFrame, price_col_prefer: str = "px_ask"):
    """Lee el Excel, construye objetos ons_pro y les asigna precio."""
    required = [
        "name","empresa","curr","law","start_date","end_date",
        "payment_frequency","amortization_dates","amortizations","rate"
    ]
    raw = pd.read_excel(path_or_bytes, dtype=str)
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    bonds = []
    errors = []
    for _, r in raw.iterrows():
        try:
            name  = str(r["name"]).strip()
            emp   = str(r["empresa"]).strip()
            curr  = str(r["curr"]).strip()
            law   = str(r["law"]).strip()
            start = parse_date_cell(r["start_date"])
            end   = parse_date_cell(r["end_date"])

            pay_freq_raw = parse_float_cell(r["payment_frequency"])
            if pd.isna(pay_freq_raw) or pay_freq_raw <= 0:
                raise ValueError(f"{name}: payment_frequency inválido -> {r['payment_frequency']}")
            pay_freq = int(round(pay_freq_raw))

            am_dates = parse_date_list(r["amortization_dates"])
            am_amts  = parse_amorts(r["amortizations"])
            if len(am_dates) != len(am_amts):
                if len(am_dates) == 1 and len(am_amts) == 0:
                    am_amts = [100.0]
                elif len(am_dates) == 0 and len(am_amts) == 1:
                    am_dates = [end.strftime("%Y-%m-%d")]
                else:
                    raise ValueError(f"{name}: inconsistencia amortizaciones {am_dates} vs {am_amts}")

            rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))
            price    = get_price_for_symbol(df_all, name, prefer=price_col_prefer)

            b = ons_pro(
                name=name, empresa=emp, curr=curr, law=law,
                start_date=start, end_date=end, payment_frequency=pay_freq,
                amortization_dates=am_dates, amortizations=am_amts,
                rate=rate_pct, price=price
            )
            bonds.append(b)
        except Exception as e:
            errors.append(f"{r.get('name','?')}: {e}")
    if errors:
        st.warning("Algunos bonos no se pudieron cargar:\n- " + "\n- ".join(errors))
    return bonds

# =====================================
# UI
# =====================================

st.title("📈 ONs & Bonos Soberanos (ARG)")

# Botón actualizar precios
col_header = st.columns([1, 1, 6])
with col_header[0]:
    if st.button("🔄 Actualizar precios", type="primary", help="Refresca precios de data912", key="refresh_prices"):
        st.cache_data.clear()
        st.rerun()

# Carga de precios y excel
with st.spinner("Cargando precios"):
    df_all = build_df_all()
    if df_all.empty:
        st.error("No hay precios disponibles")
        st.stop()
    try:
        excel_bytes = io.BytesIO(fetch_excel_bytes(EXCEL_URL_DEFAULT))
    except Exception as e:
        st.error(f"No pude descargar el Excel desde la URL: {e}")
        st.stop()
    bonds = load_ons_from_excel(excel_bytes, df_all, price_col_prefer="px_ask")
    if not bonds:
        st.error("El Excel no produjo bonos válidos.")
        st.stop()

# =========================
# Instanciar soberanos/ARGENTs y BCRA (precios desde df_all)
# =========================

def _px(sym, prefer="px_bid"):
    return get_price_for_symbol(df_all, sym, prefer=prefer)

LAW_NY = "NY"
LAW_AR = "AR"

# === GD ===
gd_29 = bond_calculator_pro(
    name="GD29", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2029,7,9), payment_frequency=6,
    amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09"],
    amortizations=[10]*10, rate=1, price=_px("GD29D"),
    fr=2, step_up_dates=[], step_up=[], outstanding=2635
)
gd_30 = bond_calculator_pro(
    name="GD30", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2030,7,9), payment_frequency=6,
    amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09"],
    amortizations=[4,8,8,8,8,8,8,8,8,8,8,8,8],
    rate=0.125, price=_px("GD30D"),
    fr=2, step_up_dates=["2021-07-09","2023-07-09","2027-07-09"],
    step_up=[0.005,0.0075,0.0175], outstanding=16000
)
gd_35 = bond_calculator_pro(
    name="GD35", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2035,7,9), payment_frequency=6,
    amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09"],
    amortizations=[10]*10, rate=0.125, price=_px("GD35D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
    step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05], outstanding=20501
)
gd_38 = bond_calculator_pro(
    name="GD38", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2038,1,9), payment_frequency=6,
    amortization_dates=[
        "2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09",
        "2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09",
        "2037-07-09","2038-01-09"
    ],
    amortizations=[4.54]*32 + [4.55]*12, rate=0.125, price=_px("GD38D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
    step_up=[0.0020,0.03875,0.0425,0.05], outstanding=20501
)
gd_41 = bond_calculator_pro(
    name="GD41", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2041,7,9), payment_frequency=6,
    amortization_dates=[
        "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09","2032-07-09",
        "2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09","2037-07-09",
        "2038-01-09","2038-07-09","2039-01-09","2039-07-09","2040-01-09","2040-07-09","2041-01-09","2041-07-09"
    ],
    amortizations=[100/28.0]*28, rate=0.125, price=_px("GD41D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
    step_up=[0.0250,0.0350,0.04875], outstanding=20501
)
gd_46 = bond_calculator_pro(
    name="GD46", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2046,7,9), payment_frequency=6,
    amortization_dates=[
        "2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09",
        "2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09",
        "2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09","2037-07-09","2038-01-09","2038-07-09","2039-01-09","2039-07-09",
        "2040-01-09","2040-07-09","2041-01-09","2041-07-09","2042-01-09","2042-07-09","2043-01-09","2043-07-09","2044-01-09","2044-07-09",
        "2045-01-09","2045-07-09","2046-01-09","2046-07-09"
    ],
    amortizations=[100/44.0]*44, rate=0.00125, price=_px("GD46D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
    step_up=[0.01125,0.0150,0.03625,0.04125,0.04375,0.05], outstanding=20501
)

# === AL / AE ===
al_29 = bond_calculator_pro(
    name="AL29", emisor="Tesoro Nacional", curr="USD", law=LAW_AR,
    start_date=datetime(2021,1,9), end_date=datetime(2029,7,9), payment_frequency=6,
    amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09"],
    amortizations=[10]*10, rate=1, price=_px("AL29D"),
    fr=2, step_up_dates=[], step_up=[], outstanding=2635
)
al_30 = bond_calculator_pro(
    name="AL30", emisor="Tesoro Nacional", curr="USD", law=LAW_AR,
    start_date=datetime(2021,1,9), end_date=datetime(2030,7,9), payment_frequency=6,
    amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09"],
    amortizations=[4,8,8,8,8,8,8,8,8,8,8,8,8], rate=0.125, price=_px("AL30D"),
    fr=2, step_up_dates=["2021-07-09","2023-07-09","2027-07-09"],
    step_up=[0.005,0.0075,0.0175], outstanding=16000
)
al_35 = bond_calculator_pro(
    name="AL35", emisor="Tesoro Nacional", curr="USD", law=LAW_AR,
    start_date=datetime(2021,1,9), end_date=datetime(2035,7,9), payment_frequency=6,
    amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09"],
    amortizations=[10]*10, rate=0.125, price=_px("AL35D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
    step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05], outstanding=20501
)
ae_38 = bond_calculator_pro(
    name="AE38", emisor="Tesoro Nacional", curr="USD", law=LAW_NY,
    start_date=datetime(2021,1,9), end_date=datetime(2038,1,9), payment_frequency=6,
    amortization_dates=[
        "2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09",
        "2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09",
        "2037-07-09","2038-01-09"
    ],
    amortizations=[4.54]*32 + [4.55]*12, rate=0.125, price=_px("AE38D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
    step_up=[0.0020,0.03875,0.0425,0.05], outstanding=20501
)
al_41 = bond_calculator_pro(
    name="AL41", emisor="Tesoro Nacional", curr="USD", law=LAW_AR,
    start_date=datetime(2021,1,9), end_date=datetime(2041,7,9), payment_frequency=6,
    amortization_dates=[
        "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09","2032-07-09",
        "2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09","2037-07-09",
        "2038-01-09","2038-07-09","2039-01-09","2039-07-09","2040-01-09","2040-07-09","2041-01-09","2041-07-09"
    ],
    amortizations=[100/28.0]*28, rate=0.125, price=_px("AL41D"),
    fr=2, step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
    step_up=[0.0250,0.0350,0.04875], outstanding=20501
)

# === BCRA (BPB/BPC/BPD) ===
bpb7d = bond_calculator_pro(
    name="BPB7", emisor="BCRA", curr="USD", law=LAW_AR,
    start_date=datetime(2024,4,30), end_date=datetime(2026,4,30), payment_frequency=6,
    amortization_dates=["2026-04-30"], amortizations=[100],
    rate=5, price=_px("BPB7D"), fr=2, step_up_dates=[], step_up=[], outstanding=966
)
bpc7d = bond_calculator_pro(
    name="BPC7", emisor="BCRA", curr="USD", law=LAW_AR,
    start_date=datetime(2024,4,30), end_date=datetime(2027,4,30), payment_frequency=6,
    amortization_dates=["2027-04-30"], amortizations=[100],
    rate=5, price=_px("BPC7D"), fr=2, step_up_dates=[], step_up=[], outstanding=966
)
bpd7d = bond_calculator_pro(
    name="BPD7", emisor="BCRA", curr="USD", law=LAW_AR,
    start_date=datetime(2024,4,30), end_date=datetime(2027,10,30), payment_frequency=6,
    amortization_dates=["2027-04-30","2027-10-30"], amortizations=[50,50],
    rate=5, price=_px("BPD7D"), fr=2, step_up_dates=[], step_up=[], outstanding=966
)

sovereigns = [gd_29, gd_30, gd_35, gd_38, gd_41, gd_46,
              al_29, al_30, al_35, ae_38, al_41,
              bpb7d, bpc7d, bpd7d]

# =========================
# Recalcular df_metrics con ambos tipos de objetos
# =========================

rows_ons = [_obj_metrics_row(b) for b in bonds]
rows_sov = [_obj_metrics_row(b) for b in sovereigns]
df_metrics = pd.DataFrame(rows_ons + rows_sov, columns=_metrics_columns()).reset_index(drop=True)

# Redondeos/formatos homogéneos
for c in ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]:
    df_metrics[c] = pd.to_numeric(df_metrics[c], errors="coerce")
df_metrics["Cupón"] = df_metrics["Cupón"].round(4)
df_metrics["Precio"] = df_metrics["Precio"].round(2)
for c in ["Yield","TNA_180","Current Yield","Paridad (%)"]:
    df_metrics[c] = df_metrics[c].round(2)
for c in ["Dur","MD","Conv"]:
    df_metrics[c] = df_metrics[c].round(2)

# =========================
# Filtros (en form para evitar rerun continuo)
# =========================

with st.form("filters"):
    fc = st.columns(3)
    with fc[0]:
        emp_opts = sorted(df_metrics["Emisor"].dropna().unique().tolist())
        sel_emp = st.multiselect("Emisor", emp_opts, default=emp_opts, key="filter_emp")
    with fc[1]:
        mon_opts = sorted(df_metrics["Moneda de Pago"].dropna().unique().tolist())
        sel_mon = st.multiselect("Moneda de Pago", mon_opts, default=mon_opts, key="filter_mon")
    with fc[2]:
        ley_opts = sorted(df_metrics["Ley"].dropna().unique().tolist())
        sel_ley = st.multiselect("Ley", ley_opts, default=ley_opts, key="filter_ley")
    submitted = st.form_submit_button("Aplicar filtros")

if not submitted:
    sel_emp = emp_opts
    sel_mon = mon_opts
    sel_ley = ley_opts

mask = (
    df_metrics["Emisor"].isin(sel_emp) &
    df_metrics["Moneda de Pago"].isin(sel_mon) &
    df_metrics["Ley"].isin(sel_ley)
)
df_view = df_metrics.loc[mask].reset_index(drop=True)

# 🔹 Definición de columnas numéricas
num_cols = ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]

# Copia segura para formateo de la grilla
dfv = df_metrics.copy()
for c in num_cols:
    dfv[c] = pd.to_numeric(dfv[c], errors="coerce")

# Muestra grilla de métricas
st.dataframe(
    dfv,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Cupón":         st.column_config.NumberColumn("Cupón",         format="%.2f"),
        "Precio":        st.column_config.NumberColumn("Precio",        format="%.2f"),
        "Yield":         st.column_config.NumberColumn("Yield",         format="%.2f"),
        "TNA_180":       st.column_config.NumberColumn("TNA_180",       format="%.2f"),
        "Dur":           st.column_config.NumberColumn("Dur",           format="%.2f"),
        "MD":            st.column_config.NumberColumn("MD",            format="%.2f"),
        "Conv":          st.column_config.NumberColumn("Conv",          format="%.2f"),
        "Current Yield": st.column_config.NumberColumn("Current Yield", format="%.2f"),
        "Paridad (%)":   st.column_config.NumberColumn("Paridad (%)",   format="%.2f"),
    },
)

# Botón para descargar CSV filtrado
csv = df_view.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Descargar CSV", data=csv, file_name="ons_metrics.csv", mime="text/csv")

st.markdown("---")

# =========================
# Flujos escalados
# =========================

colA, colB = st.columns([2, 3])
with colA:
    st.subheader("Flujos")
    tickers = ["(ninguno)"] + df_view["Ticker"].dropna().unique().tolist()
    pick = st.selectbox("Ticker", tickers, index=0, key="flow_ticker")

    mode = st.radio(
        "Modo de cálculo",
        ["Por nominales (VN)", "Por monto / precio manual"],
        horizontal=False,
        key="flow_mode",
    )

    if mode == "Por nominales (VN)":
        vn = st.number_input("Nominales (VN)", min_value=0.0, value=100.0, step=100.0, key="vn_input")
        precio_manual = None
        monto = None
    else:
        monto = st.number_input("Monto a invertir", min_value=0.0, value=10000.0, step=1000.0, key="monto_input")
        precio_manual = st.number_input(
            "Precio manual (por 100 nominal, clean)",
            min_value=0.0001, value=100.0, step=0.5, key="precio_manual_flows"
        )
        vn = None

with colB:
    if pick and pick != "(ninguno)":
        bmap = {b.name: b for b in bonds}
        bmap.update({b.name: b for b in sovereigns})
        if pick in bmap:
            b = bmap[pick]
            st.write(
                f"**{pick}** — Emisor: {getattr(b,'emisor', getattr(b,'empresa','-'))} · "
                f"Moneda: {getattr(b,'curr','-')} · Ley: {getattr(b,'law','-')} · "
                f"Cupón base: {round(float(b.rate)*100,4)}% · Precio: {float(getattr(b,'price',float('nan'))):.2f}"
            )
            df_flows = bond_flows_frame(b)

            # Escala (excluyendo fila 0)
            if mode == "Por nominales (VN)":
                scale = (vn or 0.0) / 100.0
            else:
                scale = 0.0 if not monto or not precio_manual else (monto / precio_manual)

            df_cash = df_flows.copy()
            if scale > 0:
                df_cash.loc[1:, ["Residual","Amortización","Cupón","Flujo"]] = \
                    df_cash.loc[1:, ["Residual","Amortización","Cupón","Flujo"]].astype(float) * scale

            st.dataframe(df_cash.iloc[1:].reset_index(drop=True), use_container_width=True, height=360)

            if scale > 0:
                total_cobros = float(df_cash["Flujo"].iloc[1:].sum())
                st.metric("Total a cobrar (sumatoria flujos futuros)", f"{total_cobros:,.2f}")
        else:
            st.warning(f"No encontré el bono {pick} en la lista cargada.")

st.markdown("---")

# =========================
# Métricas con precio manual
# =========================

st.subheader("Calculadora de métricas")
colM1, colM2, colM3, colM4 = st.columns([2, 1.2, 1.2, 3])

with colM1: st.markdown("**Ticker**")
with colM2: st.markdown("**Precio manual**")
with colM3: st.markdown("** **")
with colM4: st.markdown("**Resultado**")

with colM1:
    tick2 = st.selectbox(
        label="Ticker",
        options=["(ninguno)"] + df_metrics["Ticker"].dropna().unique().tolist(),
        index=0,
        key="manual_ticker",
        label_visibility="collapsed",
    )

with colM2:
    pman = st.number_input(
        label="Precio manual",
        min_value=0.0, value=100.0, step=0.5,
        key="manual_price",
        label_visibility="collapsed",
    )

with colM3:
    go_btn = st.button("Calcular métricas", key="calc_metrics_btn", type="primary", use_container_width=True)

with colM4:
    if go_btn and tick2 and tick2 != "(ninguno)":
        bmap = {b.name: b for b in bonds}
        bmap.update({b.name: b for b in sovereigns})
        if tick2 in bmap:
            b0 = bmap[tick2]
            # clonar con precio manual
            if isinstance(b0, bond_calculator_pro):
                b = bond_calculator_pro(
                    name=b0.name, start_date=b0.start_date, end_date=b0.end_date,
                    payment_frequency=b0.payment_frequency,
                    amortization_dates=b0.amortization_dates, amortizations=b0.amortizations,
                    rate=b0.rate*100.0,  # acepta en % o decimal; le paso %
                    price=pman, fr=b0.frequency,
                    step_up_dates=b0.step_up_dates, step_up=b0.step_up,
                    outstanding=b0.outstanding,
                    emisor=b0.emisor, curr=b0.curr, law=b0.law
                )
            else:
                # ons_pro
                b = ons_pro(
                    name=b0.name, empresa=getattr(b0,"empresa", getattr(b0,"emisor","-")),
                    curr=getattr(b0,"curr","-"), law=getattr(b0,"law","-"),
                    start_date=b0.start_date, end_date=b0.end_date, payment_frequency=b0.payment_frequency,
                    amortization_dates=b0.amortization_dates, amortizations=b0.amortizations,
                    rate=b0.rate*100.0, price=pman
                )
            df_one = pd.DataFrame([_obj_metrics_row(b)], columns=_metrics_columns())
            st.dataframe(df_one, use_container_width=True, height=140, hide_index=True)
        else:
            st.warning(f"No encontré el bono {tick2} para el cálculo manual.")
