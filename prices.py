# app.py
import io
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta

# =========================
# Config Streamlit
# =========================
st.set_page_config(page_title="Bonos HD", page_icon="💵", layout="wide")

# =========================
# Clase bond_calculator_pro
# =========================
class bond_calculator_pro:
    """
    Bonos con cupón fijo y amortizaciones discretas (en % de 100),
    con step-ups opcionales. Genera fechas hacia atrás desde el vencimiento
    (limitadas por la fecha de emisión), y devuelve en orden cronológico
    (primero T+1 settlement).
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
        self.frequency = max(1, int(round(12 / self.payment_frequency)))  # cupones/año

        # Normalización de amortizaciones
        self.amortization_dates = [str(d) for d in amortization_dates]
        self.amortizations = [float(a) for a in amortizations]
        if len(self.amortization_dates) != len(self.amortizations):
            raise ValueError(f"{name}: amortization dates and amounts length mismatch.")
        am_sum = float(np.nansum(self.amortizations))
        if am_sum > 100 + 1e-9:
            raise ValueError(f"{name}: amortizations sum to {am_sum:.6f} > 100.")
        self._am_sum = am_sum

        # rate: acepta % o decimal, guarda decimal
        self.rate = float(rate) / 100.0 if float(rate) >= 1 else float(rate)
        self.price = float(price)

        # Step-ups en decimal y ordenados
        steps = []
        for d, r in zip(step_up_dates, step_up):
            r_dec = float(r) if float(r) < 1 else float(r) / 100.0
            steps.append((str(d), r_dec))
        steps.sort(key=lambda x: x[0])
        self.step_up_dates = [d for d, _ in steps]
        self.step_up = [r for _, r in steps]

        self.outstanding = float(outstanding)

        # Cache
        self._cache: Dict[str, Any] = {}

    # ---- helpers internos ----
    def _settlement(self, settlement: Optional[datetime] = None) -> datetime:
        return settlement if settlement else (datetime.today() + timedelta(days=1))

    def _clear_cache(self):
        self._cache.clear()

    # ---- calendario hacia atrás ----
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

    # ---- perfil de capital ----
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

    # ---- step-ups y cupones ----
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

    # ---- flujos (precio excluido en la UI de simulación) ----
    def cash_flow(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("cash", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        caps = self.amortization_payments(settlement)
        cpns = self.coupon_payments(settlement)
        cfs = [-self.price] + [c + a for c, a in zip(cpns[1:], caps[1:])]
        self._cache[key] = cfs
        return cfs

    # ---- PV / IRR ----
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
                return round(r_new * 100.0, 1)
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
                return round(m * 100.0, 1)
            if flo * fm <= 0:
                hi, fhi = m, fm
            else:
                lo, flo = m, fm
        return round(0.5 * (lo + hi) * 100.0, 1)

    # ---- analytics ----
    def tna_180(self, settlement: Optional[datetime] = None) -> float:
        irr = self.xirr(settlement) / 100.0
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 1)

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
        return round(mac, 1)

    def modified_duration(self, settlement: Optional[datetime] = None) -> float:
        dur = self.duration(settlement)
        irr = self.xirr(settlement) / 100.0
        if not np.isfinite(dur) or not np.isfinite(irr):
            return float("nan")
        return round(dur / (1 + irr / self.frequency), 1)

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
        return round(cx, 1)

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
        return round(annual / self.price * 100.0, 1)

    # -------- helpers de período (para AI) --------
    def _last_next_coupon_dates(self, settlement: Optional[datetime] = None):
        """Devuelve (last_coupon_date, next_coupon_date) alrededor de settlement."""
        stl = self._settlement(settlement)
        # construyo las fechas del calendario completo (hacia atrás), y me quedo con la primera futura
        fut = self._schedule_backwards(settlement)
        # fut = [settlement, f1, f2, ...]; la próxima es f1 si existe
        next_coupon = fut[1] if len(fut) > 1 else None

        # para obtener la última: voy restando períodos desde next_coupon
        if next_coupon is None:
            return (None, None)
        last_coupon = next_coupon - relativedelta(months=self.payment_frequency)
        # si por edge-case quedara > settlement, retrocedo otro período
        if last_coupon > stl:
            last_coupon = last_coupon - relativedelta(months=self.payment_frequency)
        return (last_coupon, next_coupon)

    def _accrual_fraction(self, settlement: Optional[datetime] = None) -> float:
        """Fracción devengada en el período actual: días corridos / días del período (Actual/Actual por días)."""
        stl = self._settlement(settlement)
        last_cpn, next_cpn = self._last_next_coupon_dates(settlement)
        if last_cpn is None or next_cpn is None:
            return 0.0
        days_total = (next_cpn - last_cpn).days
        days_run   = max(0, (stl - last_cpn).days)
        return 0.0 if days_total <= 0 else min(1.0, days_run / days_total)

    def _period_coupon_rate_and_base(self, settlement: Optional[datetime] = None):
        """
        Devuelve (rate_period_decimal, residual_base) del período actual:
        - rate del período (con step-up) correspondiente al próximo cupón
        - residual sobre el que se calcula el cupón (residual en el inicio del período)
        """
        dates = self.generate_payment_dates(settlement)
        rates = self.step_up_rate(settlement)
        residuals = self.residual_value(settlement)
        if len(dates) < 2:
            return (0.0, 0.0)

        # Próxima fecha de pago es dates[1]
        next_date = dates[1]
        i = 1  # índice del próximo pago en nuestras listas
        rate_period = float(rates[i])  # ya en decimal
        residual_base = float(residuals[i-1])  # residual al inicio del período (en settlement)
        return (rate_period, residual_base)

    def accrued_interest(self, settlement: Optional[datetime] = None) -> float:
        """
        Interés corrido (por 100 VN) desde el último cupón hasta 'settlement'.
        Convención: Actual/Actual por días del período.
        """
        frac = self._accrual_fraction(settlement)
        if frac <= 0:
            return 0.0
        rate_period, residual_base = self._period_coupon_rate_and_base(settlement)
        coupon_full = (rate_period / self.frequency) * residual_base
        return round(coupon_full * frac, 6)  # precisión interna, luego redondeás a 1 decimal en la UI

    def parity(self, settlement: Optional[datetime] = None) -> float:
        """
        Paridad técnica = Precio (clean) / (Valor residual + AI) * 100.
        Si tu precio fuese 'dirty', usá dirty/(residual) o ajustá la fórmula según tu convención.
        """
        residual_t0 = float(self.residual_value(settlement)[0])  # por 100 VN
        ai = self.accrued_interest(settlement)                   # por 100 VN
        denom = residual_t0 + ai
        if denom <= 0 or not np.isfinite(denom):
            return float("nan")
        return round(self.price / denom * 100.0, 1)

# =========================
# Helpers de parsing Excel
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
    if pd.isna(r):
        return np.nan
    r = float(r)
    return r*100.0 if r < 1 else r

# =========================
# Fetch precios (data912)
# =========================
@st.cache_data(ttl=300)
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

@st.cache_data(ttl=300)
def load_market_data():
    url_bonds = "https://data912.com/live/arg_bonds"
    url_notes = "https://data912.com/live/arg_notes"
    url_corps = "https://data912.com/live/arg_corp"
    # url_mep = "https://data912.com/live/mep"  # si lo necesitás luego

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

    # Normalizo columnas esperadas
    # busco 'ticker' y 'bid'/'ask' comunes
    if "ticker" in df_all.columns:
        df_all["symbol"] = df_all["ticker"].astype(str)
    elif "symbol" not in df_all.columns:
        df_all["symbol"] = ""

    if "bid" in df_all.columns:
        df_all["px_bid"] = pd.to_numeric(df_all["bid"], errors="coerce")
    elif "px_bid" not in df_all.columns:
        df_all["px_bid"] = np.nan

    if "ask" in df_all.columns:
        df_all["px_ask"] = pd.to_numeric(df_all["ask"], errors="coerce")
    elif "px_ask" not in df_all.columns:
        df_all["px_ask"] = np.nan

    return df_all

def get_price_for_symbol(df_all: pd.DataFrame, name: str, prefer="px_bid") -> float:
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

# =========================
# Carga de ONs desde Excel
# =========================
@st.cache_data(ttl=600)
def load_bcp_from_excel(
    df_all: pd.DataFrame,
    adj: float = 1.0,
    price_col_prefer: str = "px_bid"
) -> list:
    url_excel_raw = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"
    content = requests.get(url_excel_raw, timeout=25).content
    raw = pd.read_excel(io.BytesIO(content), dtype=str)

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
            continue
        pay_freq = int(round(pay_freq_raw))

        am_dates = parse_date_list(r["amortization_dates"])
        am_amts  = [parse_float_cell(x) for x in str(r["amortizations"]).split(";")] if str(r["amortizations"]).strip() != "" else []
        if len(am_dates) != len(am_amts):
            if len(am_dates) == 1 and len(am_amts) == 0:
                am_amts = [100.0]
            elif len(am_dates) == 0 and len(am_amts) == 1:
                am_dates = [end.strftime("%Y-%m-%d")]
            else:
                continue

        rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))
        try:
            price    = get_price_for_symbol(df_all, name, prefer=price_col_prefer) * adj
        except Exception:
            price    = np.nan
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

# =========================
# Tabla de métricas
# =========================
def metrics_bcp(bonds: list, settlement: datetime | None = None) -> pd.DataFrame:
    rows = []
    for b in bonds:
        try:
            dates = b.generate_payment_dates(settlement)
            prox = dates[1] if len(dates) > 1 else None
            rows.append({
                "Ticker": b.name,
                "Emisor": b.emisor,
                "Ley": b.law,
                "Moneda de Pago": b.curr,
                "Precio": round(b.price, 1) if pd.notna(b.price) else np.nan,
                "TIR": round(b.xirr(settlement), 1) if pd.notna(b.price) else np.nan,
                "TNA SA": round(b.tna_180(settlement), 1) if pd.notna(b.price) else np.nan,
                "Modified Duration": round(b.modified_duration(settlement), 1) if pd.notna(b.price) else np.nan,
                "Duration": round(b.duration(settlement), 1) if pd.notna(b.price) else np.nan,
                "Convexidad": round(b.convexity(settlement), 1) if pd.notna(b.price) else np.nan,
                "Paridad": round(b.parity(settlement), 1) if pd.notna(b.price) else np.nan,
                "Calificación": b.calificacion,
                "Próxima Fecha de Pago": prox,
                "Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d"),
            })
        except Exception:
            rows.append({
                "Ticker": b.name, "Emisor": b.emisor, "Ley": b.law, "Moneda de Pago": b.curr,
                "Precio": np.nan, "TIR": np.nan, "TNA SA": np.nan, "Modified Duration": np.nan,
                "Duration": np.nan, "Convexidad": np.nan, "Paridad": np.nan,
                "Calificación": b.calificacion,
                "Próxima Fecha de Pago": None,
                "Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d"),
            })
    df = pd.DataFrame(rows)
    # 1 decimal garantizado
    num_cols = ["Precio","TIR","TNA SA","Modified Duration","Duration","Convexidad","Paridad"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    return df

def center_table(df: pd.DataFrame) -> str:
    # Render simple con HTML centrado y 1 decimal
    fmt = {c: "{:,.1f}".format for c in df.select_dtypes(include=[np.number]).columns}
    styled = df.style.format(fmt).hide(axis="index")
    html = styled.to_html()
    html = html.replace('<table', '<table style="margin-left:auto;margin-right:auto;text-align:center;"')
    html = html.replace('<th ', '<th style="text-align:center;" ')
    html = html.replace('<td ', '<td style="text-align:center;" ')
    return html

# =========================
# Manual: lista de soberanos
# =========================
def manual_bonds_factory(df_all):
    def px(sym): 
        try: return get_price_for_symbol(df_all, sym, prefer="px_bid")
        except: return np.nan

    # --- Ojo: si querés ajustar calificaciones, editá acá ---
    gd_29 = bond_calculator_pro(
        name="GD29", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2029,7,9),
        payment_frequency=6,
        amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09"],
        amortizations=[10]*10, rate=1, price=px("GD29D"),
        step_up_dates=[], step_up=[], outstanding=2635, calificacion="CCC-"
    )
    gd_30 = bond_calculator_pro(
        name="GD30", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2030,7,9),
        payment_frequency=6,
        amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09"],
        amortizations=[4]+[8]*12, rate=0.125, price=px("GD30D"),
        step_up_dates=["2021-07-09","2023-07-09","2027-07-09"], step_up=[0.005,0.0075,0.0175],
        outstanding=16000, calificacion="CCC-"
    )
    gd_35 = bond_calculator_pro(
        name="GD35", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2035,7,9),
        payment_frequency=6,
        amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09"],
        amortizations=[10]*10, rate=0.125, price=px("GD35D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05],
        outstanding=20501, calificacion="CCC-"
    )
    gd_38 = bond_calculator_pro(
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
        rate=0.125, price=px("GD38D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
        step_up=[0.0020,0.03875,0.0425,0.05],
        outstanding=20501, calificacion="CCC-"
    )
    gd_41 = bond_calculator_pro(
        name="GD41", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2041,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09",
            "2030-07-09","2031-01-09","2031-07-09","2032-01-09","2032-07-09",
            "2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09",
            "2035-07-09","2036-01-09","2036-07-09","2037-01-09","2037-07-09",
            "2038-01-09","2038-07-09","2039-01-09","2039-07-09","2040-01-09",
            "2040-07-09","2041-01-09","2041-07-09"
        ],
        amortizations=[100/28.0]*28, rate=0.125, price=px("GD41D"),
        step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
        step_up=[0.0250,0.0350,0.04875],
        outstanding=20501, calificacion="CCC-"
    )
    gd_46 = bond_calculator_pro(
        name="GD46", emisor="Tesoro Nacional", curr="CCL", law="NY",
        start_date=datetime(2021,1,9), end_date=datetime(2046,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09",
            "2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09",
            "2030-01-09","2030-07-09","2031-01-09","2031-07-09","2032-01-09",
            "2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09",
            "2035-01-09","2035-07-09","2036-01-09","2036-07-09","2037-01-09",
            "2037-07-09","2038-01-09","2038-07-09","2039-01-09","2039-07-09",
            "2040-01-09","2040-07-09","2041-01-09","2041-07-09","2042-01-09",
            "2042-07-09","2043-01-09","2043-07-09","2044-01-09","2044-07-09",
            "2045-01-09","2045-07-09","2046-01-09","2046-07-09"
        ],
        amortizations=[100/44.0]*44, rate=0.00125, price=px("GD46D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.0150,0.03625,0.04125,0.04375,0.05],
        outstanding=20501, calificacion="CCC-"
    )
    al_29 = bond_calculator_pro(
        name="AL29", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2029,7,9),
        payment_frequency=6,
        amortization_dates=["2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09"],
        amortizations=[10]*10, rate=1, price=px("AL29D"),
        step_up_dates=[], step_up=[], outstanding=2635, calificacion="CCC-"
    )
    al_30 = bond_calculator_pro(
        name="AL30", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2030,7,9),
        payment_frequency=6,
        amortization_dates=["2024-07-09","2025-01-09","2025-07-09","2026-01-09","2026-07-09","2027-01-09","2027-07-09","2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09","2030-07-09"],
        amortizations=[4]+[8]*12, rate=0.125, price=px("AL30D"),
        step_up_dates=["2021-07-09","2023-07-09","2027-07-09"], step_up=[0.005,0.0075,0.0175],
        outstanding=16000, calificacion="CCC-"
    )
    al_35 = bond_calculator_pro(
        name="AL35", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2035,7,9),
        payment_frequency=6,
        amortization_dates=["2031-01-09","2031-07-09","2032-01-09","2032-07-09","2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09","2035-07-09"],
        amortizations=[10]*10, rate=0.125, price=px("AL35D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09","2027-07-09","2028-07-09"],
        step_up=[0.01125,0.015,0.03625,0.04125,0.0475,0.05],
        outstanding=20501, calificacion="CCC-"
    )
    ae_38 = bond_calculator_pro(
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
        amortizations=[4.55]*21 + [4.45],
        rate=0.125, price=px("AE38D"),
        step_up_dates=["2021-07-09","2022-07-09","2023-07-09","2024-07-09"],
        step_up=[0.0020,0.03875,0.0425,0.05],
        outstanding=20501, calificacion="CCC-"
    )
    al_41 = bond_calculator_pro(
        name="AL41", emisor="Tesoro Nacional", curr="MEP", law="ARG",
        start_date=datetime(2021,1,9), end_date=datetime(2041,7,9),
        payment_frequency=6,
        amortization_dates=[
            "2028-01-09","2028-07-09","2029-01-09","2029-07-09","2030-01-09",
            "2030-07-09","2031-01-09","2031-07-09","2032-01-09","2032-07-09",
            "2033-01-09","2033-07-09","2034-01-09","2034-07-09","2035-01-09",
            "2035-07-09","2036-01-09","2036-07-09","2037-01-09","2037-07-09",
            "2038-01-09","2038-07-09","2039-01-09","2039-07-09","2040-01-09",
            "2040-07-09","2041-01-09","2041-07-09"
        ],
        amortizations=[100/28.0]*28, rate=0.125, price=px("AL41D"),
        step_up_dates=["2021-07-09","2022-07-09","2029-07-09"],
        step_up=[0.0250,0.0350,0.04875],
        outstanding=20501, calificacion="CCC-"
    )
    bpb7d = bond_calculator_pro(
        name="BPB7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2026,4,30),
        payment_frequency=6,
        amortization_dates=["2026-04-30"], amortizations=[100],
        rate=5, price=px("BPB7D"),
        step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    )
    bpc7d = bond_calculator_pro(
        name="BPC7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2027,4,30),
        payment_frequency=6,
        amortization_dates=["2027-04-30"], amortizations=[100],
        rate=5, price=px("BPC7D"),
        step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    )
    bpd7d = bond_calculator_pro(
        name="BPD7", emisor="BCRA", curr="CCL/MEP", law="ARG",
        start_date=datetime(2024,4,30), end_date=datetime(2027,10,30),
        payment_frequency=6,
        amortization_dates=["2027-04-30","2027-10-30"], amortizations=[50,50],
        rate=5, price=px("BPD7D"),
        step_up_dates=[], step_up=[], outstanding=966, calificacion="CCC-"
    )

    return [gd_29, gd_30, gd_35, gd_38, gd_41, gd_46,
            al_29, al_30, al_35, ae_38, al_41,
            bpb7d, bpc7d, bpd7d]

# =========================
# Simulador de flujos
# =========================
def build_cashflow_table(selected_bonds: list, mode: str, inputs: dict) -> pd.DataFrame:
    rows = []
    for b in selected_bonds:
        # SIN el primer flujo (precio): solo pagos
        dates = b.generate_payment_dates()[1:]
        flows = b.cash_flow()[1:]

        if mode == "nominal":
            nominal = float(inputs.get(b.name, 0) or 0)
        else:  # "monto"
            monto = float(inputs.get(b.name, 0) or 0)
            nominal = (monto / b.price) if (b.price and b.price == b.price) else 0.0

        for d, f in zip(dates, flows):
            rows.append({"Fecha": d, "Ticker": b.name, "Flujo": round(f * nominal, 1)})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Fecha", "Total"])
    df_total = df.groupby("Fecha", as_index=False)["Flujo"].sum()
    df_total = df_total.rename(columns={"Flujo":"Total"})
    df_total["Total"] = df_total["Total"].round(1)
    return df_total
# =========================
# Calculadora de métricas (3 bonos, precio manual)
# =========================
def clone_with_price(b: bond_calculator_pro, new_price: float) -> bond_calculator_pro:
    return bond_calculator_pro(
        name=b.name, emisor=b.emisor, curr=b.curr, law=b.law,
        start_date=b.start_date, end_date=b.end_date,
        payment_frequency=b.payment_frequency,
        amortization_dates=b.amortization_dates, amortizations=b.amortizations,
        rate=b.rate, price=new_price,
        step_up_dates=b.step_up_dates, step_up=b.step_up,
        outstanding=b.outstanding, calificacion=b.calificacion
    )

def compare_metrics_three(bond_map: Dict[str, bond_calculator_pro], sel_names: list, prices: list) -> pd.DataFrame:
    clones = []
    for n, p in zip(sel_names, prices):
        if not n:
            continue
        base = bond_map[n]
        clones.append(clone_with_price(base, float(p)))
    return metrics_bcp(clones)

# =========================
# App UI
# =========================
def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Elegí sección", ["Bonos HD", "Lecaps", "Otros"], index=0)

    # --- Carga de mercado + botón refrescar ---
    df_all = load_market_data()
    if st.sidebar.button("🔄 Actualizar precios"):
        load_market_data.clear()  # limpia cache
        df_all = load_market_data()
        st.sidebar.success("Precios actualizados.")

    # --- Construcción de universos ---
    ons_bonds = load_bcp_from_excel(df_all, adj=1.005, price_col_prefer="px_ask")
    manual_bonds = manual_bonds_factory(df_all)
    all_bonds = ons_bonds + manual_bonds
    name_to_bond = {b.name: b for b in all_bonds}

    if page == "Bonos HD":
        st.title("Bonos HD")
        st.caption("Tabla de métricas, simulador de flujos y comparador de métricas (3 bonos).")

        # =========================
        # 1) TABLA DE MÉTRICAS + FILTROS
        # =========================
        st.subheader("Métricas")
        df_full = metrics_bcp(all_bonds)

        # Filtros
        colf1, colf2, colf3 = st.columns(3)
        emisores = sorted([e for e in df_full["Emisor"].dropna().unique()])
        monedas  = sorted([m for m in df_full["Moneda de Pago"].dropna().unique()])
        leyes    = sorted([l for l in df_full["Ley"].dropna().unique()])

        with colf1:
            f_emisor = st.multiselect("Filtrar Emisor", emisores, default=emisores)
        with colf2:
            f_moneda = st.multiselect("Filtrar Moneda de Pago", monedas, default=monedas)
        with colf3:
            f_ley    = st.multiselect("Filtrar Ley", leyes, default=leyes)

        mask = (
            df_full["Emisor"].isin(f_emisor) &
            df_full["Moneda de Pago"].isin(f_moneda) &
            df_full["Ley"].isin(f_ley)
        )
        df_filtered = df_full.loc[mask].reset_index(drop=True)

        # Render centrado y 1 decimal
        html_tbl = center_table(df_filtered)
        st.markdown(html_tbl, unsafe_allow_html=True)

        st.divider()

        # =========================
        # 2) SIMULADOR DE FLUJOS
        # =========================
        st.subheader("Simulador de Flujos (sin desembolso inicial)")
        colA, colB = st.columns([1, 2])
        with colA:
            sel_bonds = st.multiselect(
                "Seleccioná bonos",
                options=sorted(name_to_bond.keys()),
                default=[]
            )
            mode = st.radio("Modo de entrada", ["nominal", "monto"], horizontal=True, index=0)
        with colB:
            inputs = {}
            if sel_bonds:
                st.write("Parámetros por bono:")
                for n in sel_bonds:
                    if mode == "nominal":
                        val = st.number_input(f"VN de {n}", min_value=0.0, step=100.0, value=0.0, key=f"vn_{n}")
                    else:
                        val = st.number_input(f"Monto (USD) para {n}", min_value=0.0, step=100.0, value=0.0, key=f"amt_{n}")
                    inputs[n] = val

        if sel_bonds:
            selected_objs = [name_to_bond[n] for n in sel_bonds]
            df_cf = build_cashflow_table(selected_objs, mode, inputs)
            st.markdown("**Flujo consolidado por fecha (USD):**")
            html_cf = center_table(df_cf)
            st.markdown(html_cf, unsafe_allow_html=True)
        else:
            st.info("Seleccioná al menos un bono para ver flujos.")

        st.divider()

        # =========================
        # 3) Calculadora de Métricas (3 bonos con precio manual)
        # =========================
        st.subheader("Comparador de Métricas (3 bonos, precio manual)")
        c1, c2, c3 = st.columns(3)
        with c1:
            b1 = st.selectbox("Bono 1", [""] + sorted(name_to_bond.keys()), index=0, key="cmp_b1")
            p1 = st.number_input("Precio 1", min_value=0.0, step=0.1, value=0.0, key="cmp_p1")
        with c2:
            b2 = st.selectbox("Bono 2", [""] + sorted(name_to_bond.keys()), index=0, key="cmp_b2")
            p2 = st.number_input("Precio 2", min_value=0.0, step=0.1, value=0.0, key="cmp_p2")
        with c3:
            b3 = st.selectbox("Bono 3", [""] + sorted(name_to_bond.keys()), index=0, key="cmp_b3")
            p3 = st.number_input("Precio 3", min_value=0.0, step=0.1, value=0.0, key="cmp_p3")

        if any([b1, b2, b3]):
            df_cmp = compare_metrics_three(name_to_bond, [b1, b2, b3], [p1, p2, p3])
            st.markdown("**Métricas comparadas (con precios manuales):**")
            st.markdown(center_table(df_cmp), unsafe_allow_html=True)
        else:
            st.info("Elegí al menos un bono y definí precios para comparar.")

    elif page == "Lecaps":
        st.title("Lecaps")
        st.info("Sección en construcción. Próximamente métricas y simuladores para Lecaps.")

    else:
        st.title("Otros")
        st.info("Sección en construcción para otros instrumentos y herramientas.")

if __name__ == "__main__":
    main()


