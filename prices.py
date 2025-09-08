# app.py
# ============================================
# Bonos HD - ONs + Soberanos (con precios live data912)
# ============================================

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re, io, os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# =========================
# ---- UI CONFIG ---------
# =========================
st.set_page_config(page_title="Bonos HD", page_icon="💹", layout="wide")

# =========================
# ---- Helpers: parsing ---
# =========================
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def parse_date_cell(s):
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
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    parts = str(cell).replace(",", "/").split(";")
    out = []
    for p in parts:
        d = parse_date_cell(p)
        out.append(d.strftime("%Y-%m-%d"))
    return out

def parse_float_cell(x):
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
    if pd.isna(r): return np.nan
    r = float(r)
    return r*100.0 if r < 1 else r

# =========================
# ---- data912 prices -----
# =========================
@st.cache_data(ttl=120)
def fetch_json(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def to_df(payload):
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes", "rows"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    return pd.json_normalize(payload)

def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize to columns: symbol, px_bid, px_ask
    Common variants: ticker/symbol, bid/bestBid, ask/bestAsk, px_bid/px_ask
    """
    df = df.copy()
    # symbol
    sym_col = None
    for c in ("symbol","ticker","code","name"):
        if c in df.columns:
            sym_col = c; break
    if sym_col is None:
        raise ValueError("No symbol/ticker column in price feed")
    df["symbol"] = df[sym_col].astype(str)

    # bid
    bid_col = None
    for c in ("px_bid","bid","bestBid","Bid"):
        if c in df.columns:
            bid_col = c; break
    if bid_col is None:
        df["px_bid"] = np.nan
    else:
        df["px_bid"] = pd.to_numeric(df[bid_col], errors="coerce")

    # ask
    ask_col = None
    for c in ("px_ask","ask","bestAsk","Ask"):
        if c in df.columns:
            ask_col = c; break
    if ask_col is None:
        df["px_ask"] = np.nan
    else:
        df["px_ask"] = pd.to_numeric(df[ask_col], errors="coerce")

    return df[["symbol","px_bid","px_ask"]].drop_duplicates()

@st.cache_data(ttl=120, show_spinner=False)
def build_df_all_from_data912() -> pd.DataFrame:
    url_bonds = "https://data912.com/live/arg_bonds"
    url_notes = "https://data912.com/live/arg_notes"
    url_corps = "https://data912.com/live/arg_corp"
    # url_mep  = "https://data912.com/live/mep"  # available if you need it

    df_bonds = normalize_price_df(to_df(fetch_json(url_bonds)))
    df_notes = normalize_price_df(to_df(fetch_json(url_notes)))
    df_corps = normalize_price_df(to_df(fetch_json(url_corps)))

    df_all = pd.concat([df_bonds, df_notes, df_corps], ignore_index=True)
    df_all = df_all.groupby("symbol", as_index=False).agg({"px_bid":"max","px_ask":"max"})
    return df_all

# ---------- price resolver ----------
def get_price_for_symbol(df_all: pd.DataFrame, name: str, prefer="px_bid") -> float:
    """
    Try exact 'name', then 'name + D' (e.g., GD30 -> GD30D).
    Falls back bid/ask as needed. Raises if not found.
    """
    def _pick(row):
        if prefer in row and pd.notna(row[prefer]): return float(row[prefer])
        alt = "px_ask" if prefer == "px_bid" else "px_bid"
        if alt in row and pd.notna(row[alt]): return float(row[alt])
        raise KeyError("no valid bid/ask")
    row = df_all.loc[df_all["symbol"] == name]
    if not row.empty:
        return _pick(row.iloc[0])
    row = df_all.loc[df_all["symbol"] == f"{name}D"]
    if not row.empty:
        return _pick(row.iloc[0])
    raise KeyError(f"Price not found for {name} (or {name}D)")

# ===============================
# ---- bond_calculator_pro ------
# ===============================
class bond_calculator_pro:
    """
    Fixed-rate bond with optional step-ups and discrete amortizations (% over 100).
    Payment dates are generated backwards from maturity (end_date) in paid
    frequency steps, limited by start_date, returned ascending with T+1 first.
    All flows are per 100 nominal.
    """
    def __init__(
        self,
        name: str,
        emisor: str,
        curr: str,
        law: str,
        start_date: datetime,
        end_date: datetime,
        payment_frequency: int,
        amortization_dates: List[str],
        amortizations: List[float],
        rate: float,
        price: float,
        step_up_dates: List[str],
        step_up: List[float],
        outstanding: float,
        calificacion: str
    ):
        self.name = str(name)
        self.emisor = str(emisor)
        self.curr = str(curr)
        self.law = str(law)
        self.start_date = start_date
        self.end_date = end_date
        self.calificacion = str(calificacion)

        pf = int(payment_frequency)
        if pf <= 0:
            raise ValueError(f"{name}: payment_frequency must be > 0 (months).")
        self.payment_frequency = pf
        self.frequency = max(1, int(round(12 / self.payment_frequency)))  # coupons/year

        self.amortization_dates = [str(d) for d in amortization_dates]
        self.amortizations = [float(a) for a in amortizations]
        if len(self.amortization_dates) != len(self.amortizations):
            raise ValueError(f"{name}: amortization dates and amounts length mismatch.")
        am_sum = float(np.nansum(self.amortizations))
        if am_sum > 100 + 1e-9:
            raise ValueError(f"{name}: amortizations sum to {am_sum:.6f} > 100.")
        self._am_sum = am_sum  # keep for diagnostics

        # rate: accept percent or decimal, store decimal
        self.rate = float(rate) / 100.0 if float(rate) >= 1 else float(rate)
        self.price = float(price)

        steps = []
        for d, r in zip(step_up_dates, step_up):
            r_dec = float(r) if float(r) < 1 else float(r) / 100.0
            steps.append((str(d), r_dec))
        steps.sort(key=lambda x: x[0])
        self.step_up_dates = [d for d, _ in steps]
        self.step_up = [r for _, r in steps]

        self.outstanding = float(outstanding)
        self._cache: Dict[str, Any] = {}

    # internals
    def _settlement(self, settlement: Optional[datetime] = None) -> datetime:
        return settlement if settlement else (datetime.today() + timedelta(days=1))

    def _schedule_backwards(self, settlement: Optional[datetime] = None):
        key = ("sched", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        stl = self._settlement(settlement)
        back = [self.end_date]
        cur = self.end_date
        while True:
            prev = cur - relativedelta(months=self.payment_frequency)
            if prev <= self.start_date:
                break
            back.append(prev)
            cur = prev
        future = sorted(d for d in back if d > stl)
        out = [stl] + future
        self._cache[key] = out
        return out

    def generate_payment_dates(self, settlement: Optional[datetime] = None) -> List[str]:
        return [d.strftime("%Y-%m-%d") for d in self._schedule_backwards(settlement)]

    # capital & coupons
    def residual_value(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("residual", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        dates = self.generate_payment_dates(settlement)
        amap = {d: float(a) for d, a in zip(self.amortization_dates, self.amortizations)}
        res = []
        R = 100.0
        for d in dates:
            if d in amap:
                R = max(0.0, R - amap[d])
            res.append(R)
        self._cache[key] = res
        return res

    def amortization_payments(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("amorts", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        dates = self.generate_payment_dates(settlement)
        amap = {d: float(a) for d, a in zip(self.amortization_dates, self.amortizations)}
        out = [amap.get(d, 0.0) for d in dates]
        self._cache[key] = out
        return out

    def step_up_rate(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("steprate", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        dates = self.generate_payment_dates(settlement)
        if not self.step_up_dates:
            out = [self.rate] * len(dates)
            self._cache[key] = out
            return out
        out = []
        for s in dates:
            sdt = datetime.strptime(s, "%Y-%m-%d")
            r = self.rate
            for d_str, r_step in zip(self.step_up_dates, self.step_up):
                if sdt >= datetime.strptime(d_str, "%Y-%m-%d"):
                    r = r_step
                else:
                    break
            out.append(r)
        self._cache[key] = out
        return out

    def coupon_payments(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("coupons", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        rates = self.step_up_rate(settlement)
        residuals = self.residual_value(settlement)
        cpns = [0.0]
        f = self.frequency
        for i in range(1, len(rates)):
            cpns.append((rates[i] / f) * residuals[i - 1])
        self._cache[key] = cpns
        return cpns

    def cash_flow(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("cash", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        caps = self.amortization_payments(settlement)
        cpns = self.coupon_payments(settlement)
        cfs = [-self.price] + [c + a for c, a in zip(cpns[1:], caps[1:])]
        self._cache[key] = cfs
        return cfs

    # times & PV/IRR
    def _times_years(self, settlement: Optional[datetime] = None) -> np.ndarray:
        key = ("times", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        stl = self._settlement(settlement)
        dates = [datetime.strptime(s, "%Y-%m-%d") for s in self.generate_payment_dates(settlement)]
        t = np.array([(d - stl).days / 365.0 for d in dates], dtype=float)
        self._cache[key] = t
        return t

    def _xnpv(self, r: float, settlement: Optional[datetime] = None) -> float:
        t = self._times_years(settlement)
        c = np.array(self.cash_flow(settlement), dtype=float)
        with np.errstate(over="ignore", invalid="ignore"):
            disc = (1.0 + r) ** t
            return float(np.sum(c / disc))

    def xnpv(self, rate_custom: float = 0.08, settlement: Optional[datetime] = None) -> float:
        return self._xnpv(rate_custom, settlement)

    def xirr(self, settlement: Optional[datetime] = None) -> float:
        # Newton + fallback bisection
        def f(r): return self._xnpv(r, settlement)
        def df(r):
            t = self._times_years(settlement)
            c = np.array(self.cash_flow(settlement), dtype=float)
            with np.errstate(over="ignore", invalid="ignore"):
                return float(np.sum(-t * c / (1.0 + r) ** (t + 1)))

        r = 0.2
        for _ in range(12):
            fr, dfr = f(r), df(r)
            if not np.isfinite(fr) or not np.isfinite(dfr) or dfr == 0:
                break
            r_new = r - fr / dfr
            if r_new <= -0.999:
                r_new = (r - 0.999) * 0.5
            if abs(r_new - r) < 1e-10:
                return round(r_new * 100.0, 2)
            r = r_new

        lo, hi = -0.95, 5.0
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)):
            return float("nan")
        if flo * fhi > 0:
            for hi_try in (10.0, 20.0):
                fhi = f(hi_try)
                if flo * fhi <= 0:
                    hi = hi_try
                    break
            else:
                return float("nan")
        for _ in range(60):
            m = 0.5 * (lo + hi)
            fm = f(m)
            if abs(fm) < 1e-12:
                return round(m * 100.0, 2)
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return round(0.5 * (lo + hi) * 100.0, 2)

    # analytics
    def tna_180(self, settlement: Optional[datetime] = None) -> float:
        irr = self.xirr(settlement) / 100.0
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 2)

    def duration(self, settlement: Optional[datetime] = None) -> float:
        irr = self.xirr(settlement) / 100.0
        if not np.isfinite(irr):
            return float("nan")
        t = self._times_years(settlement)
        c = np.array(self.cash_flow(settlement), dtype=float)
        disc = (1 + irr / self.frequency) ** (self.frequency * t)
        pv_flows = c / disc
        pv_flows[0] = 0.0
        pv = pv_flows.sum()
        if pv <= 0 or np.isnan(pv):
            return float("nan")
        mac = float((t * pv_flows).sum() / pv)
        return round(mac, 2)

    def modified_duration(self, settlement: Optional[datetime] = None) -> float:
        dur = self.duration(settlement)
        irr = self.xirr(settlement) / 100.0
        if not np.isfinite(dur) or not np.isfinite(irr):
            return float("nan")
        return round(dur / (1 + irr / self.frequency), 2)

    def convexity(self, settlement: Optional[datetime] = None) -> float:
        irr = self.xirr(settlement) / 100.0
        if not np.isfinite(irr):
            return float("nan")
        t = self._times_years(settlement)
        c = np.array(self.cash_flow(settlement), dtype=float)
        disc = (1 + irr / self.frequency) ** (self.frequency * t)
        pv = c / disc
        pv[0] = 0.0
        price_pos = pv.sum()
        if price_pos <= 0 or np.isnan(price_pos):
            return float("nan")
        cx_term = (c * t * (t + 1 / self.frequency)) / ((1 + irr / self.frequency) ** (self.frequency * t + 2))
        cx = float(cx_term[1:].sum() / price_pos)
        return round(cx, 2)

    def current_yield(self, settlement: Optional[datetime] = None) -> float:
        cpns = self.coupon_payments(settlement)
        dates = [datetime.strptime(s, "%Y-%m-%d") for s in self.generate_payment_dates(settlement)]
        stl = self._settlement(settlement)
        idx = [i for i, d in enumerate(dates) if d > stl and cpns[i] > 0]
        if not idx:
            return float("nan")
        i0 = idx[0]
        n = min(self.frequency, len(cpns) - i0)
        annual = float(sum(cpns[i0:i0 + n]))
        return round(annual / self.price * 100.0, 2)

    def parity(self, settlement: Optional[datetime] = None) -> float:
        vt = float(self.residual_value(settlement)[0])
        return float("nan") if vt == 0 else round(self.price / vt * 100.0, 2)

    def to_row(self, settlement: Optional[datetime] = None) -> Dict[str, Any]:
        dates = self.generate_payment_dates(settlement)
        prox = dates[1] if len(dates) > 1 else None
        return {
            "Ticker": self.name,
            "Emisor": self.emisor,
            "Moneda de Pago": self.curr,
            "Ley": self.law,
            "Precio": round(self.price, 2),
            "TIR": self.xirr(settlement),
            "TNA SA": self.tna_180(settlement),
            "Duration": self.duration(settlement),
            "Modified Duration": self.modified_duration(settlement),
            "Convexidad": self.convexity(settlement),
            "Paridad": self.parity(settlement),
            "Calificación": self.calificacion,
            "Próxima Fecha de Pago": prox,
            "Fecha de Vencimiento": self.end_date.strftime("%Y-%m-%d"),
        }

# ===================================
# ---- Excel -> bond objects --------
# ===================================
def load_bcp_from_excel(
    excel_path_or_file,
    df_all: pd.DataFrame,
    adj: float = 1.0,
    price_col_prefer: str = "px_bid"
) -> list[bond_calculator_pro]:
    raw = pd.read_excel(excel_path_or_file, dtype=str)

    required = ["name","empresa","curr","law","start_date","end_date",
                "payment_frequency","amortization_dates","amortizations",
                "rate","outstanding","calificación"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

    out = []
    for _, r in raw.iterrows():
        name  = str(r["name"]).strip()
        emisor = str(r["empresa"]).strip()
        curr  = str(r["curr"]).strip()
        law   = str(r["law"]).strip()
        start = parse_date_cell(r["start_date"])
        end   = parse_date_cell(r["end_date"])

        pay_freq_raw = parse_float_cell(r["payment_frequency"])
        if pd.isna(pay_freq_raw) or pay_freq_raw <= 0:
            raise ValueError(f"{name}: payment_frequency inválido -> {r['payment_frequency']}")
        pay_freq = int(round(pay_freq_raw))

        am_dates = parse_date_list(r["amortization_dates"])
        am_amts  = [parse_float_cell(x) for x in str(r["amortizations"]).split(";")] if str(r["amortizations"]).strip() != "" else []
        if len(am_dates) != len(am_amts):
            if len(am_dates) == 1 and len(am_amts) == 0:
                am_amts = [100.0]
            elif len(am_dates) == 0 and len(am_amts) == 1:
                am_dates = [end.strftime("%Y-%m-%d")]
            else:
                raise ValueError(f"{name}: amortization mismatch {am_dates} vs {am_amts}")

        rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))
        price    = get_price_for_symbol(df_all, name, prefer=price_col_prefer) * adj
        outstanding = parse_float_cell(r["outstanding"])
        calif = str(r["calificación"]).strip()

        b = bond_calculator_pro(
            name=name, emisor=emisor, curr=curr, law=law,
            start_date=start, end_date=end, payment_frequency=pay_freq,
            amortization_dates=am_dates, amortizations=am_amts,
            rate=rate_pct, price=price,
            step_up_dates=[], step_up=[],
            outstanding=outstanding, calificacion=calif
        )
        out.append(b)
    return out

def metrics_bcp(bonds: list[bond_calculator_pro], settlement: datetime | None = None) -> pd.DataFrame:
    rows = []
    for b in bonds:
        rows.append(b.to_row(settlement))
    df = pd.DataFrame(rows)
    # numeric formatting
    for c in ["TIR","TNA SA","Paridad"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    for c in ["Duration","Modified Duration","Convexidad","Precio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    # center align
    df = df.style.set_properties(**{"text-align":"center"}).set_table_styles(
        [dict(selector="th", props=[("text-align","center")])]
    )
    return df

# =====================================================
# ---- Manual sovereigns (NY/ARG) using df_all prices -
# =====================================================
def build_manual_bonds(df_all: pd.DataFrame) -> list[bond_calculator_pro]:
    def p(sym): return get_price_for_symbol(df_all, sym, "px_bid")
    objs = []

    # --- GD29
    objs.append(bond_calculator_pro(
        name="GD29", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2029,7,9),
        payment_frequency=6,
        amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09",
                            "2027-01-09","2027-07-09","2028-01-09","2028-07-09",
                            "2029-01-09","2029-07-09"],
        amortizations=[10]*10,
        rate=1, price=p("GD29D"), step_up_dates=[], step_up=[],
        outstanding=2635, calificacion="CCC-"
    ))

    # --- GD30
    objs.append(bond_calculator_pro(
        name="GD30", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2030,7,9),
        payment_frequency=6,
        amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09",
                            "2027-01-09","2027-07-09","2028-01-09","2028-07-09",
                            "2029-01-09","2029-07-09","2030-01-09","2030-07-09"],
        amortizations=[4]+[8]*12,
        rate=0.125, price=p("GD30D"),
        step_up_dates=["2021-07-09","2023-07-09","2027-07-09"],
        step_up=[0.005,0.0075,0.0175],
        outstanding=16000, calificacion="CCC-"
    ))

    # --- GD35
    objs.append(bond_calculator_pro(
        name="GD35", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2035,7,9),
        payment_frequency=6,
        amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09",
                            "2033-01-09","2033-07-09","2034-01-09","2034-07-09",
                            "2035-01-09","2035-07-09"],
        amortizations=[10]*10,
        rate=0.125, price=p("GD35D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05],
        outstanding=20501, calificacion="CCC-"
    ))

    # --- GD38
    objs.append(bond_calculator_pro(
        name="GD38", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2038,1,9),
        payment_frequency=6,
        amortization_dates=[
            "2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09",
            "2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09",
            "2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09",
            "2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09",
            "2037-07-09","2038-01-09"
        ],
        amortizations=[4.55]*21 + [4.45],
        rate=0.125, price=p("GD38D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
        step_up=[0.0020,0.03875,0.0425,0.05],
        outstanding=20501, calificacion="CCC-"
    ))

    # --- GD41
    objs.append(bond_calculator_pro(
        name="GD41", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2041,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09",
            "2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09",
            "2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09",
            "2037-01-09","2037-07-09","2038-01-09","2038-07-09","2039-01-09","2039-07-09",
            "2040-01-09","2040-07-09","2041-01-09","2041-07-09"
        ],
        amortizations=[100/28.0]*28,
        rate=0.125, price=p("GD41D"),
        step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
        step_up=[0.0250,0.0350,0.04875],
        outstanding=20501, calificacion="CCC-"
    ))

    # --- GD46
    objs.append(bond_calculator_pro(
        name="GD46", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2046,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09",
            "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09",
            "2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09",
            "2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09",
            "2037-01-09","2037-07-09","2038-01-09","2038-07-09","2039-01-09","2039-07-09",
            "2040-01-09","2040-07-09","2041-01-09","2041-07-09","2042-01-09","2042-07-09",
            "2043-01-09","2043-07-09","2044-01-09","2044-07-09","2045-01-09","2045-07-09",
            "2046-01-09","2046-07-09"
        ],
        amortizations=[100/44.0]*44,
        rate=0.00125, price=p("GD46D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.0150,0.03625,0.04125,0.04375,0.05],
        outstanding=20501, calificacion="CCC-"
    ))

    # ARGENTs
    objs.append(bond_calculator_pro(
        name="AL29", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2029,7,9),
        payment_frequency=6,
        amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09"],
        amortizations=[10]*10, rate=1, price=p("AL29D"),
        step_up_dates=[], step_up=[], outstanding=2635, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="AL30", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2030,7,9),
        payment_frequency=6,
        amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09",
                            "2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09",
                            "2029-07-09","2030-01-09","2030-07-09"],
        amortizations=[4]+[8]*12, rate=0.125, price=p("AL30D"),
        step_up_dates=["2021-07-09","2023-07-09","2027-07-09"],
        step_up=[0.005,0.0075,0.0175], outstanding=16000, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="AL35", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2035,7,9),
        payment_frequency=6,
        amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09"],
        amortizations=[10]*10, rate=0.125, price=p("AL35D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05], outstanding=20501, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="AE38", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2038,1,9),
        payment_frequency=6,
        amortization_dates=[
            "2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09",
            "2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09",
            "2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09",
            "2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09",
            "2037-07-09","2038-01-09"
        ],
        amortizations=[4.55]*21 + [4.45], rate=0.125, price=p("AE38D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
        step_up=[0.0020,0.03875,0.0425,0.05], outstanding=20501, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="AL41", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2041,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09",
            "2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09",
            "2034-01-09","2034-07-09","2035-01-09","2035-07-09","2036-01-09","2036-07-09",
            "2037-01-09","2037-07-09","2038-01-09","2038-07-09","2039-01-09","2039-07-09",
            "2040-01-09","2040-07-09","2041-01-09","2041-07-09"
        ],
        amortizations=[100/28.0]*28, rate=0.125, price=p("AL41D"),
        step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
        step_up=[0.0250,0.0350,0.04875], outstanding=20501, calificacion="CCC-"
    ))

    # BCRA
    objs.append(bond_calculator_pro(
        name="BPB7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2026,4,30),
        payment_frequency=6, amortization_dates=["2026-04-30"], amortizations=[100],
        rate=5, price=p("BPB7D"), step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="BPC7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2027,4,30),
        payment_frequency=6, amortization_dates=["2027-04-30"], amortizations=[100],
        rate=5, price=p("BPC7D"), step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    ))
    objs.append(bond_calculator_pro(
        name="BPD7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2027,10,30),
        payment_frequency=6, amortization_dates=["2027-04-30","2027-10-30"], amortizations=[50,50],
        rate=5, price=p("BPD7D"), step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    ))

    return objs

# ==========================================
# ---- Cash flow aggregation for selection --
# ==========================================
def combined_cash_flows(selected: list[bond_calculator_pro], settlement: datetime | None = None) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(columns=["date","cash_flow"])
    # collect all dates
    all_dates = set()
    per = []
    for b in selected:
        d = b.generate_payment_dates(settlement)
        cf = b.cash_flow(settlement)
        per.append(pd.DataFrame({"date": d, "cash_flow": cf, "Ticker": b.name}))
        all_dates.update(d)
    all_dates = sorted(all_dates)
    # sum by date
    agg = pd.DataFrame({"date": all_dates})
    agg["cash_flow"] = 0.0
    for df in per:
        agg = agg.merge(df[["date","cash_flow"]], on="date", how="left", suffixes=("","_tmp"))
        agg["cash_flow"] = agg["cash_flow"] + agg["cash_flow_tmp"].fillna(0.0)
        agg.drop(columns=["cash_flow_tmp"], inplace=True)
    return agg

# ==========================================
# ---- Price overrides for comparator -------
# ==========================================
def metrics_for_with_price(b: bond_calculator_pro, price_override: float | None, settlement: datetime | None):
    orig = b.price
    try:
        if price_override is not None and np.isfinite(price_override) and price_override > 0:
            b.price = float(price_override)
        return b.to_row(settlement)
    finally:
        b.price = orig

# =========================
# ---- Sidebar (Nav) ------
# =========================
st.sidebar.title("Navegación")
section = st.sidebar.radio(
    "Elegí el módulo",
    ["Bonos HD", "Lecaps (próx.)", "Otras métricas (próx.)"],
    index=0
)

# =========================
# ---- Data bootstrap -----
# =========================
with st.spinner("Cargando precios live..."):
    df_all = build_df_all_from_data912()

# Excel path / upload (para ONs)
st.sidebar.markdown("---")
st.sidebar.subheader("Listado ONs (Excel)")
default_path = r"C:\Users\mmarzano\Documents\Modelos - Calculadora\listado_ons.xlsx"
excel_path = st.sidebar.text_input("Ruta local", value=default_path)
excel_upload = st.sidebar.file_uploader("o subí el Excel", type=["xlsx"])

price_adj = st.sidebar.number_input("Ajuste de precio (x)", value=1.0, min_value=0.5, max_value=1.5, step=0.005, help="Multiplicador aplicado a precios para ONs")

# build objects
errors = []
ons_bonds = []
try:
    src = excel_upload if excel_upload is not None else (excel_path if os.path.exists(excel_path) else None)
    if src is not None:
        ons_bonds = load_bcp_from_excel(src, df_all, adj=price_adj, price_col_prefer="px_bid")
    else:
        st.sidebar.warning("No se encontró el Excel. Subí el archivo o corrige la ruta.")
except Exception as e:
    errors.append(f"Excel ONs: {e}")

manual_bonds = []
try:
    manual_bonds = build_manual_bonds(df_all)
except Exception as e:
    errors.append(f"Soberanos manuales: {e}")

all_bonds = ons_bonds + manual_bonds
bond_map = {b.name: b for b in all_bonds}

if errors:
    st.sidebar.error(" / ".join(errors))

# =========================
# ---- BONOS HD (Main) ----
# =========================
if section == "Bonos HD":
    st.title("Bonos HD")
    # internal dropdown (subsections)
    sub = st.selectbox(
        "Ir a:",
        ["Resumen y precios", "Flujos simulados", "Comparador de métricas"],
        index=0
    )

    if sub == "Resumen y precios":
        st.subheader("Tabla de métricas (ONs + Soberanos)")
        settlement = None  # usa T+1 default
        df_metrics = metrics_bcp(all_bonds, settlement)
        st.dataframe(df_metrics, use_container_width=True)

    elif sub == "Flujos simulados":
        st.subheader("Simulación de Cash Flows (suma de seleccionados)")
        # picker
        names = sorted([b.name for b in all_bonds])
        picks = st.multiselect("Elegí los bonos", names)
        sel = [bond_map[n] for n in picks] if picks else []
        if sel:
            agg = combined_cash_flows(sel, settlement=None)
            # center align display
            styled = agg.style.format({"cash_flow":"{:.2f}"}).set_properties(**{"text-align":"center"}).set_table_styles(
                [dict(selector="th", props=[("text-align","center")])]
            )
            st.dataframe(styled, use_container_width=True)
        else:
            st.info("Seleccioná al menos un bono para ver el flujo combinado.")

    elif sub == "Comparador de métricas":
        st.subheader("Comparador de 3 bonos (con precio editable)")
        cols = st.columns(3)
        picks = []
        price_inputs = []
        for i, c in enumerate(cols):
            with c:
                sel = st.selectbox(f"Ticker #{i+1}", ["(ninguno)"] + sorted(bond_map.keys()), index=0, key=f"cmp_{i}")
                if sel != "(ninguno)":
                    b = bond_map[sel]
                    price_val = st.number_input(f"Precio {sel}", value=float(b.price), min_value=0.01, step=0.01, key=f"price_{i}")
                    picks.append(b)
                    price_inputs.append(price_val)
        if picks:
            rows = []
            for b, p in zip(picks, price_inputs):
                rows.append(metrics_for_with_price(b, p, settlement=None))
            df_cmp = pd.DataFrame(rows)[
                ["Ticker","Emisor","Ley","Moneda de Pago","Precio","TIR","TNA SA",
                 "Duration","Modified Duration","Convexidad","Paridad","Calificación",
                 "Próxima Fecha de Pago","Fecha de Vencimiento"]
            ]
            df_cmp = df_cmp.style.set_properties(**{"text-align":"center"}).set_table_styles(
                [dict(selector="th", props=[("text-align","center")])]
            )
            st.dataframe(df_cmp, use_container_width=True)
        else:
            st.info("Elegí hasta 3 bonos para comparar.")

# =========================
# ---- Placeholders -------
# =========================
elif section == "Lecaps (próx.)":
    st.title("Lecaps")
    st.info("Sección en construcción. Acá vas a poder ver curvas, pricing y métricas específicas de LECAPs.")

elif section == "Otras métricas (próx.)":
    st.title("Otras métricas")
    st.info("Sección en construcción. Podrás agregar herramientas y visualizaciones adicionales aquí.")


