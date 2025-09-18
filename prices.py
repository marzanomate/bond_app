# app.py
import io
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
import QuantLib as ql
from scipy import optimize
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from requests.adapters import HTTPAdapter, Retry


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
    (limitadas por la fecha de emisión) y devuelve en orden cronológico
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
        self.frequency = max(1, int(round(12 / self.payment_frequency)))  # cupones/año (entero)

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
        self._cache: Dict[str, Any] = {}

    # ----------------- helpers internos -----------------
    def _settlement(self, settlement: Optional[datetime] = None) -> datetime:
        return settlement if settlement else (datetime.today() + timedelta(days=1))

    def _as_dt(self, d):
        return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")

    def _amort_map_dt(self) -> Dict[datetime, float]:
        """Amorts con clave datetime (para comparar por fechas)."""
        return {self._as_dt(d): float(a) for d, a in zip(self.amortization_dates, self.amortizations)}

    def _clear_cache(self):
        self._cache.clear()

    # ----------------- calendario hacia atrás -----------------
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
        out = [stl] + future  # t0 = settlement
        self._cache[key] = out
        return out

    def generate_payment_dates(self, settlement: Optional[datetime] = None) -> List[str]:
        return [d.strftime("%Y-%m-%d") for d in self._schedule_backwards(settlement)]

    # ----------------- saldo técnico (corrige amortizaciones pasadas) -----------------
    def outstanding_on(self, ref_date: Optional[datetime] = None) -> float:
        """
        Principal outstanding (por 100) luego de amortizaciones con fecha <= ref_date.
        """
        if ref_date is None:
            ref_date = self._settlement()
        ref_date = self._as_dt(ref_date)
        paid = sum(a for d, a in zip(self.amortization_dates, self.amortizations)
                   if self._as_dt(d) <= ref_date)
        return max(0.0, 100.0 - paid)

    # ----------------- perfil de capital -----------------
    def residual_value(self, settlement: Optional[datetime] = None) -> List[float]:
        """
        residual[i] = saldo técnico en la fecha 'dates[i]' (post amort de esa fecha si aplica).
        Esto hace que amortizaciones anteriores (ya ocurridas) se reflejen en t0 y siguientes.
        """
        key = ("residual", (settlement or 0))
        if key in self._cache:
            return self._cache[key]

        dates_dt = [self._as_dt(s) for s in self.generate_payment_dates(settlement)]
        res = [self.outstanding_on(d) for d in dates_dt]
        self._cache[key] = res
        return res

    def amortization_payments(self, settlement: Optional[datetime] = None) -> List[float]:
        """
        cap[0] = 0 en settlement; cap[i] = amort programada en esa fecha futura (0 si no hay).
        (Las amortizaciones anteriores a settlement NO aparecen acá porque ya quedaron
        incorporadas en el saldo técnico).
        """
        key = ("amorts", (settlement or 0))
        if key in self._cache:
            return self._cache[key]

        am_dt = self._amort_map_dt()
        dates_dt = [self._as_dt(s) for s in self.generate_payment_dates(settlement)]
        caps = [0.0]  # en t0 no hay pago
        for d in dates_dt[1:]:
            caps.append(am_dt.get(d, 0.0))

        self._cache[key] = caps
        return caps

    # ----------------- step-ups y cupones -----------------
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
        """
        Cupón en t_i = (tasa del período (t_{i-1}, t_i]) * (saldo al inicio del período) / frecuencia.
        - Saldo al inicio del período = outstanding_on(t_{i-1})  (ya neto de amort en t_{i-1})
        - Tasa del período: usamos la que aplica a t_i (o equivalentemente a (t_{i-1}, t_i]).
        """
        key = ("coupons", (settlement or 0))
        if key in self._cache:
            return self._cache[key]

        dates_dt = [self._as_dt(s) for s in self.generate_payment_dates(settlement)]
        rates = self.step_up_rate(settlement)
        f = self.frequency

        cpns = [0.0]  # en t0 no hay cupón
        for i in range(1, len(dates_dt)):
            period_start = dates_dt[i-1]  # t_{i-1} (settlement o cupón previo)
            rate_interval = float(rates[i])  # tasa correspondiente al período que finaliza en t_i
            base = self.outstanding_on(period_start)
            cpns.append((rate_interval / f) * base)

        self._cache[key] = cpns
        return cpns

    # ----------------- flujos -----------------
    def cash_flow(self, settlement: Optional[datetime] = None) -> List[float]:
        key = ("cash", (settlement or 0))
        if key in self._cache:
            return self._cache[key]
        caps = self.amortization_payments(settlement)
        cpns = self.coupon_payments(settlement)
        cfs = [-self.price] + [c + a for c, a in zip(cpns[1:], caps[1:])]
        self._cache[key] = cfs
        return cfs

    # ----------------- PV / IRR -----------------
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

    # ----------------- analytics -----------------
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

        # ----------------- intereses corridos y paridad -----------------
    def _last_next_coupon_dates(self, settlement: Optional[datetime] = None):
        """Devuelve (última fecha de cupón, próxima fecha de cupón) alrededor de settlement."""
        stl = self._settlement(settlement)
        sched = self._schedule_backwards(settlement)
        if len(sched) < 2:
            return (None, None)
        next_coupon = sched[1]
        last_coupon = next_coupon - relativedelta(months=self.payment_frequency)
        if last_coupon > stl:
            last_coupon = last_coupon - relativedelta(months=self.payment_frequency)
        return (last_coupon, next_coupon)

    def _accrual_fraction(self, settlement: Optional[datetime] = None) -> float:
        """Fracción devengada en el período actual (Actual/Actual)."""
        stl = self._settlement(settlement)
        last_cpn, next_cpn = self._last_next_coupon_dates(settlement)
        if last_cpn is None or next_cpn is None:
            return 0.0
        days_total = (next_cpn - last_cpn).days
        days_run = max(0, (stl - last_cpn).days)
        return 0.0 if days_total <= 0 else min(1.0, days_run / days_total)

    def _period_coupon_rate_and_base(self, settlement: Optional[datetime] = None):
        """
        Retorna (tasa del período, base residual al inicio del período).
        Usamos la tasa aplicable al próximo cupón y el saldo vigente al inicio.
        """
        dates = self.generate_payment_dates(settlement)
        rates = self.step_up_rate(settlement)
        residuals = self.residual_value(settlement)
        if len(dates) < 2:
            return (0.0, 0.0)
        rate_period = float(rates[1])
        residual_base = float(residuals[0])
        return (rate_period, residual_base)

    def accrued_interest(self, settlement: Optional[datetime] = None) -> float:
        """Interés corrido (por 100 VN)."""
        frac = self._accrual_fraction(settlement)
        if frac <= 0:
            return 0.0
        rate_period, residual_base = self._period_coupon_rate_and_base(settlement)
        coupon_full = (rate_period / self.frequency) * residual_base
        return round(coupon_full * frac, 6)

    def parity(self, settlement: Optional[datetime] = None) -> float:
        """
        Paridad técnica = Precio clean / (Residual técnico + AI) * 100.
        """
        residual_t0 = float(self.residual_value(settlement)[0])
        ai = self.accrued_interest(settlement)
        denom = residual_t0 + ai
        if denom <= 0 or not np.isfinite(denom):
            return float("nan")
        return round(self.price / denom * 100.0, 1)

    def current_yield(self, settlement: Optional[datetime] = None) -> float:

        stl = self._settlement(settlement)
        dates = [datetime.strptime(s, "%Y-%m-%d") for s in self.generate_payment_dates(settlement)]
        if len(dates) < 2:
            return float("nan")

        # Próxima fecha de cupón
        next_coupon = dates[1]
        prev_coupon = dates[0]

        # Base de cálculo: residual al inicio del período
        residuals = self.residual_value(settlement)
        base = residuals[0]

        # Tasa del período (considerando step-ups)
        rates = self.step_up_rate(settlement)
        r = rates[1]  # tasa vigente para este período

        # Cupón del próximo período (por 100 VN)
        coupon_next = (r / self.frequency) * base

        # Anualizar: multiplicar por frecuencia
        coupon_annual = coupon_next * self.frequency

        # Current Yield sobre precio de mercado
        return round((coupon_annual / self.price) * 100.0, 1) 

    # ==========================
    # Precio <-> Rendimiento
    # ==========================
    def price_from_irr(self, irr_pct: float, settlement: Optional[datetime] = None) -> float:
        """
        Dada una TIR efectiva anual en %, calcula el precio clean que la implica
        para este bono (descontando todos los flujos futuros a esa tasa).
        """
        try:
            r = float(irr_pct) / 100.0
            if not np.isfinite(r):
                return float("nan")
            t = self._times_years(settlement)
            c = np.array(self.cash_flow(settlement), dtype=float)
            c[0] = 0.0  # ignorar el flujo inicial negativo (precio), lo vamos a calcular
            disc = (1.0 + r) ** t
            price = float(np.sum(c / disc))
            return round(price, 2)
        except Exception:
            return float("nan")

    def yield_from_price(self, price_override: float, settlement: Optional[datetime] = None) -> float:
        """
        Dado un precio clean, devuelve la TIR efectiva anual (%) del bono.
        (No deja mutado el objeto.)
        """
        old = self.price
        try:
            self.price = float(price_override)
            y = self.xirr(settlement)
            return float(y)
        finally:
            self.price = old

# --------------------------------------------------------
# LECAPs/BONCAPS
# --------------------------------------------------------

class lecaps:
    def __init__(self, name, start_date, end_date, tem, price):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.tem = tem              # TEM en decimal (p.ej. 0.0235 = 2.35% mensual efectiva)
        self.price = price          # precio clean por 100 VN
        self.settlement = datetime.today() + timedelta(days=1)
        self.calendar = ql.Argentina(ql.Argentina.Merval)
        self.convention = ql.Following

    def _adjust_next_business_day(self, dt: datetime) -> datetime:
        qd = ql.Date(dt.day, dt.month, dt.year)
        ad = self.calendar.adjust(qd, self.convention)  # roll forward si feriado/fin de semana
        return datetime(ad.year(), int(ad.month()), ad.dayOfMonth())

    def generate_payment_dates(self):
        d_settle = self._adjust_next_business_day(self.settlement)
        d_mty    = self._adjust_next_business_day(self.end_date)
        return [d_settle.strftime("%Y-%m-%d"), d_mty.strftime("%Y-%m-%d")]

    def _months_30_360(self) -> float:
        """Meses 30/360 (Bond Basis) entre start_date y end_date."""
        dc = ql.Thirty360(ql.Thirty360.BondBasis)
        ql_start = ql.Date(self.start_date.day, self.start_date.month, self.start_date.year)
        ql_end   = ql.Date(self.end_date.day,   self.end_date.month,   self.end_date.year)
        days = dc.dayCount(ql_start, ql_end)
        return days / 30.0

    def _years_act365_from_settlement(self) -> float:
        """Años ACT/365 desde settlement a maturity (para descontar con TIR)."""
        d0 = self._adjust_next_business_day(self.settlement)
        dm = self._adjust_next_business_day(self.end_date)
        return max(0.0, (dm - d0).days / 365.0)

    def cash_flow(self):
        payments = []
        capital = 100.0

        months = self._months_30_360()
        interests = (1.0 + self.tem) ** months - 1.0
        final_payment = capital * (1.0 + interests)

        payments.append(-float(self.price))
        payments.append(float(final_payment))
        return payments

    def xnpv(self, dates=None, cash_flow=None, rate_custom=0.08):
        dates = self.generate_payment_dates()
        dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates]
        cash_flow = self.cash_flow()
        d0 = self.settlement
        return sum(
            cf / (1.0 + rate_custom) ** ((dt - d0).days / 365.0)
            for cf, dt in zip(cash_flow, dates)
        )

    def xirr(self):
        # TIR efectiva anual en %
        try:
            result = optimize.newton(lambda r: self.xnpv(rate_custom=r), 0.0)
        except Exception:
            result = optimize.brentq(lambda r: self.xnpv(rate_custom=r), -0.99, 1e10)
        return round(result * 100.0, 2)

    def tem_from_irr(self):
        irr = self.xirr() / 100.0            # EA (decimal)
        tem = (1.0 + irr) ** (30.0 / 365.0) - 1.0
        return round(tem * 100.0, 2)

    def tna30(self):
        tem_dec = ((1.0 + (self.xirr() / 100.0)) ** (30.0 / 365.0)) - 1.0
        tna30 = tem_dec * 12.0
        return round(tna30 * 100.0, 2)

    def duration(self):
        # Macaulay con 2 flujos (precio inicial y pago final)
        dates = self.generate_payment_dates()
        dates = [datetime.strptime(date_str, '%Y-%m-%d') for date_str in dates]
        cash_flow = self.cash_flow()
        irr = self.xirr() / 100.0
        d0 = self.settlement
        # evitar división por cero
        if not np.isfinite(irr):
            return float("nan")
        # PV del bono (sin el flujo t0 negativo)
        pv = sum(
            cf / (1.0 + irr) ** ((dt - d0).days / 365.0)
            for i, (cf, dt) in enumerate(zip(cash_flow, dates)) if i > 0
        )
        if pv <= 0 or not np.isfinite(pv):
            return float("nan")
        mac = sum(
            ((dt - d0).days / 365.0) * (cf / (1.0 + irr) ** ((dt - d0).days / 365.0))
            for i, (cf, dt) in enumerate(zip(cash_flow, dates)) if i > 0
        ) / pv
        return round(mac, 2)

    def modified_duration(self):
        dur = self.duration()
        irr = self.xirr() / 100.0
        if not np.isfinite(dur) or not np.isfinite(irr):
            return float("nan")
        return round(dur / (1.0 + irr), 2)

    # ======================
    # NUEVO 1: Retorno Directo
    # ======================
    def direct_return(self):
        """
        Retorno Directo = (1 + TIR)^Dur - 1  (devuelto en %).
        Usa TIR efectiva anual y Duration (años).
        """
        try:
            irr = self.xirr() / 100.0
            dur = self.duration()
            if not np.isfinite(irr) or not np.isfinite(dur):
                return float("nan")
            dr = (1.0 + irr) ** float(dur) - 1.0
            return round(dr * 100.0, 2)
        except Exception:
            return float("nan")

    # ==========================
    # NUEVO 2: Price from IRR EA
    # ==========================
    def price_from_irr(self, irr_pct: float) -> float:
        """
        Dado una TIR efectiva anual en %, devuelve el precio (clean) que la
        implica para este instrumento (2 flujos).
        """
        try:
            t_years = self._years_act365_from_settlement()
            if t_years <= 0:
                return float("nan")
            # Pago final según TEM y 30/360
            months = self._months_30_360()
            final_payment = 100.0 * ((1.0 + self.tem) ** months)
            r = float(irr_pct) / 100.0
            price = final_payment / ((1.0 + r) ** t_years)
            return round(price, 2)
        except Exception:
            return float("nan")

    def yield_from_price(self, price_override: float) -> float:
        """
        Dado un precio clean, devuelve la TIR efectiva anual (%) de la LECAP/BONCAP.
        No deja mutado el objeto.
        """
        old = self.price
        try:
            self.price = float(price_override)
            return float(self.xirr())
        finally:
            self.price = old

    # (opcional) helper para setear directamente el precio a partir de una TIR:
    def set_price_from_irr(self, irr_pct: float) -> float:
        p = self.price_from_irr(irr_pct)
        if np.isfinite(p):
            self.price = p
        return p


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

@st.cache_data(ttl=1200)  # 1200 segundos = 20 minutos
def load_market_data():
    url_bonds = "https://data912.com/live/arg_bonds"
    url_notes = "https://data912.com/live/arg_notes"
    url_corps = "https://data912.com/live/arg_corp"
    url_mep = "https://data912.com/live/mep"

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

    df_bonds = to_df(fetch_json(url_bonds)); df_bonds["source"] = "bonds"
    df_notes = to_df(fetch_json(url_notes)); df_notes["source"] = "notes"
    df_corps = to_df(fetch_json(url_corps)); df_corps["source"] = "corps"
    df_mep   = to_df(fetch_json(url_mep));   df_mep["source"]   = "mep"

    df_all = pd.concat([df_bonds, df_notes, df_corps], ignore_index=True, sort=False)
    return df_all, df_mep

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

# ----------------------------------------------------------------
# Construyo enviroment para LECAPs/Boncaps
# ----------------------------------------------------------------

# --- Normalizador mínimo para df_all (NO cambia tus funciones de búsqueda) ---
def normalize_market_df(df_all: pd.DataFrame) -> pd.DataFrame:
    df = df_all.copy()
    ren = {}
    if "ticker" in df.columns: ren["ticker"] = "symbol"
    if "ask" in df.columns:    ren["ask"]    = "px_ask"
    if "bid" in df.columns:    ren["bid"]    = "px_bid"
    df = df.rename(columns=ren)
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].astype(str).str.strip().str.upper()
    for c in ("px_ask","px_bid"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _get_ask_price(df_all: pd.DataFrame, ticker: str) -> float:
    if df_all is None or df_all.empty:
        return np.nan

    t = str(ticker).strip().upper()

    # buscá por symbol; si no existe esa col, probá 'ticker'
    col_sym = "symbol" if "symbol" in df_all.columns else ("ticker" if "ticker" in df_all.columns else None)
    if col_sym is None:
        return np.nan

    row = df_all.loc[df_all[col_sym].astype(str).str.strip().str.upper() == t]
    if row.empty:
        return np.nan

    # columnas posibles de ask/bid
    ask_cols = [c for c in ("px_ask", "ask") if c in df_all.columns]
    bid_cols = [c for c in ("px_bid", "bid") if c in df_all.columns]

    val = np.nan
    for c in ask_cols:
        v = row[c].iloc[0]
        if pd.notna(v):
            val = float(v)
            break
    if pd.isna(val):
        for c in bid_cols:
            v = row[c].iloc[0]
            if pd.notna(v):
                val = float(v)
                break

    return val if pd.notna(val) else np.nan

def build_lecaps_metrics(rows, df_all, today=None):
    import pandas as pd, numpy as np

    # --- spec ---
    df_spec = pd.DataFrame(rows, columns=["Ticker","Vencimiento","Emision","TEM_str","Tipo"])
    df_spec["Ticker"] = df_spec["Ticker"].astype(str).str.strip().str.upper()
    df_spec["Vencimiento"] = pd.to_datetime(df_spec["Vencimiento"], dayfirst=True, errors="coerce")
    df_spec["Emision"]     = pd.to_datetime(df_spec["Emision"],     dayfirst=True, errors="coerce")

    # TEM (mensual) a decimal
    def to_tem_dec(x):
        if pd.isna(x): return np.nan
        if isinstance(x, str):
            try: x = float(x.replace("%","").replace(",","."))
            except Exception: return np.nan
        return float(x)/100.0
    df_spec["TEM_dec"] = df_spec["TEM_str"].apply(to_tem_dec)

    # --- traer precio ask / bid ---
    mkt = normalize_market_df(df_all)  # <- usa el normalizador de arriba
    if "symbol" in mkt.columns:
        px_df = mkt[["symbol"] + [c for c in ("px_ask","px_bid") if c in mkt.columns]].copy()
        precio = px_df.get("px_ask", pd.Series(np.nan, index=px_df.index))
        if "px_bid" in px_df.columns:
            precio = precio.fillna(px_df["px_bid"])
        # >>> ajuste ask*1.005 (o bid si fue fallback)
        precio = precio * 1.0
        px_df = pd.DataFrame({"Ticker": px_df["symbol"], "Precio": precio})
        df_spec = df_spec.merge(px_df, on="Ticker", how="left")
    else:
        df_spec["Precio"] = np.nan

    # --- métricas con tu clase 'lecaps' ---
    out = []
    for _, r in df_spec.iterrows():
        try:
            if any(pd.isna(r[k]) for k in ["Vencimiento","Emision","TEM_dec","Precio"]):
                raise ValueError("inputs incompletos")

            obj = lecaps(
                name=r["Ticker"],
                start_date=r["Emision"].to_pydatetime(),
                end_date=r["Vencimiento"].to_pydatetime(),
                tem=float(r["TEM_dec"]),   # mensual (decimal)
                price=float(r["Precio"])   # precio clean
            )

            # helpers seguros
            def safe(fn):
                try: return fn()
                except Exception: return np.nan

            tirea = safe(obj.xirr)              # % EA
            dur   = safe(obj.duration)          # años
            md    = safe(obj.modified_duration)
            tna30 = safe(obj.tna30)             # %
            tem_i = safe(obj.tem_from_irr)      # % mensual

            direct = ((1 + (tirea or 0)/100.0)**(dur or 0) - 1.0)*100.0 if pd.notna(tirea) and pd.notna(dur) else np.nan

            out.append({
                "Ticker": r["Ticker"],
                "Tipo": r["Tipo"],
                "Vencimiento": r["Vencimiento"].date().strftime("%d/%m/%Y"),
                "Precio": round(float(r["Precio"]), 2),
                "Rendimiento (TIR EA)": round(tirea, 2) if pd.notna(tirea) else np.nan,
                "Retorno Directo": round(direct, 2) if pd.notna(direct) else np.nan,
                "TNA 30": round(tna30, 2) if pd.notna(tna30) else np.nan,
                "TEM": round(tem_i, 2) if pd.notna(tem_i) else np.nan,
                "Duration": round(dur, 2) if pd.notna(dur) else np.nan,
                "Modified Duration": round(md, 2) if pd.notna(md) else np.nan,
            })
        except Exception:
            out.append({
                "Ticker": r.get("Ticker"),
                "Tipo": r.get("Tipo"),
                "Vencimiento": r["Vencimiento"].date().strftime("%d/%m/%Y") if pd.notna(r.get("Vencimiento")) else "",
                "Precio": np.nan,
                "Rendimiento (TIR EA)": np.nan,
                "Retorno Directo": np.nan,
                "TNA 30": np.nan,
                "TEM": np.nan,
                "Duration": np.nan,
                "Modified Duration": np.nan,
            })

    cols = [
        "Ticker","Tipo","Vencimiento","Precio",
        "Rendimiento (TIR EA)","Retorno Directo","TNA 30","TEM",
        "Duration","Modified Duration"
    ]
    return pd.DataFrame(out)[cols]

def build_lecaps_objects(rows, df_all_norm) -> dict[str, lecaps]:
    """
    Crea objetos 'lecaps' a partir de LECAPS_ROWS y precios de mercado
    usando ASK con fallback a BID, multiplicado por 1.005.
    Retorna un dict: {ticker: lecaps_obj}
    """
    # armo un map rápido de {symbol: precio_ask/bid}
    px_df = df_all_norm.copy()
    if "symbol" not in px_df.columns:
        # intentar normalizar por las dudas
        px_df = normalize_market_df(px_df)

    # precio = ask con fallback en bid
    precio = px_df.get("px_ask", pd.Series(np.nan, index=px_df.index))
    if "px_bid" in px_df.columns:
        precio = precio.fillna(px_df["px_bid"])
    px_map = dict(zip(px_df["symbol"].astype(str).str.upper(), (precio * 1.005).astype(float)))

    le_map = {}
    for (ticker, vto, emi, tem, tipo) in rows:
        try:
            price = float(px_map.get(str(ticker).strip().upper(), np.nan))
            if not np.isfinite(price):
                continue
            obj = lecaps(
                name=ticker,
                start_date=parse_date_cell(emi),
                end_date=parse_date_cell(vto),
                tem=float(tem)/100.0 if float(tem) >= 1 else float(tem),
                price=price
            )
            le_map[ticker] = obj
        except Exception:
            # si algo falla con un ticker, lo salteamos
            pass
    return le_map

def build_lecaps_table(spec_rows: list, df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Versión simple que usa build_lecaps_metrics y devuelve el DF listo para mostrar.
    (Sin argumento 'today' para evitar el error.)
    """
    df = build_lecaps_metrics(spec_rows, df_all)
    # Asegurar redondeo uniforme a 2 decimales en numéricos:
    for c in ["Precio","Rendimiento (TIR EA)","","TNA 30","TEM","Duration","Modified Duration"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(2)
    return df

# =========================
# Carga de ONs desde Excel
# =========================

@st.cache_resource(show_spinner=False)
def load_bcp_from_excel(df_all: pd.DataFrame, adj: float = 1.0, price_col_prefer: str = "px_bid") -> list:
    url_excel_raw = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"
    try:
        r = requests.get(url_excel_raw, timeout=25)
        r.raise_for_status()
        content = r.content
    except Exception as e:
        raise RuntimeError(f"No se pudo descargar el Excel de ONs: {e}")

    # sanity-check: archivos .xlsx son ZIP y empiezan con 'PK'
    if not content.startswith(b"PK"):
        raise RuntimeError("El contenido descargado no parece un .xlsx (posible rate-limit de GitHub o URL incorrecta).")

    try:
        raw = pd.read_excel(io.BytesIO(content), dtype=str, engine="openpyxl")
    except Exception as e:
        raise RuntimeError(f"No se pudo abrir el Excel de ONs: {e}")

    required = [
        "name","empresa","curr","law","start_date","end_date",
        "payment_frequency","amortization_dates","amortizations",
        "rate","outstanding","calificación"
    ]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns in Excel: {missing}")

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
    stl = settlement
    for b in bonds:
        try:
            dates = b.generate_payment_dates(stl)
            prox = dates[1] if len(dates) > 1 else None
            rows.append({
                "Ticker": b.name,
                "Emisor": b.emisor,
                "Ley": b.law,
                "Moneda de Pago": b.curr,
                "Precio": round(b.price, 1),
                "TIR": b.xirr(stl),
                "TNA SA": b.tna_180(stl),
                "Modified Duration": b.modified_duration(stl),
                "Duration": b.duration(stl),
                "Convexidad": b.convexity(stl),
                "Paridad": b.parity(stl),
                "Current Yield": b.current_yield(stl),   # <-- NUEVO
                "Calificación": b.calificacion,
                "Próxima Fecha de Pago": prox,
                "Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d"),
            })
        except Exception as e:
            rows.append({
                "Ticker": getattr(b, "name", np.nan),
                "Emisor": getattr(b, "emisor", np.nan),
                "Ley": getattr(b, "law", np.nan),
                "Moneda de Pago": getattr(b, "curr", np.nan),
                "Precio": round(getattr(b, "price", np.nan), 1) if hasattr(b, "price") else np.nan,
                "TIR": np.nan, "TNA SA": np.nan, "Modified Duration": np.nan,
                "Duration": np.nan, "Convexidad": np.nan, "Paridad": np.nan,
                "Current Yield": np.nan,                  # <-- NUEVO
                "Calificación": getattr(b, "calificacion", np.nan),
                "Próxima Fecha de Pago": None,
                "Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d") if hasattr(b, "end_date") else None,
            })
            print(f"⚠️ {getattr(b, 'name', '?')}: {e}")

    df = pd.DataFrame(rows)

    # formateo numérico a 1 decimal
    for c in ["TIR","TNA SA","Paridad","Current Yield"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
    for c in ["Duration","Modified Duration","Convexidad","Precio"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").round(1)

    return df.reset_index(drop=True)

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

@st.cache_resource(show_spinner=False)
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
    # =========================
    # BA7DD
    # =========================
    ba7dd = bond_calculator_pro(
        name="BA7DD", emisor="Provincia Buenos Aires", curr="CCL/MEP", law="NY",
        start_date=datetime(2021,6,30), end_date=datetime(2037,9,1),
        payment_frequency=6,
        amortization_dates=[
            "2028-09-01","2029-03-01","2029-09-01",   # 0.75%
            "2030-03-01","2030-09-01",                 # 6.15%
            "2031-03-01","2031-09-01","2032-03-01","2032-09-01","2033-03-01","2033-09-01", # 6.35%
            "2034-03-01","2034-09-01","2035-03-01","2035-09-01","2036-03-01","2036-09-01", # 5.90%
            "2037-03-01",  # 5.98%
            "2037-09-01"   # 5.97%
        ],
        amortizations=[
            0.75,0.75,0.75,
            6.15,6.15,
            6.35,6.35,6.35,6.35,6.35,6.35,
            5.90,5.90,5.90,5.90,5.90,5.90,
            5.98,
            5.97
        ],
        rate=2.0,   # arranca 2% anual
        price=px("BA7DD"),  # función tuya px() para buscar precio
        step_up_dates=["2021-09-01","2022-09-01","2023-09-01","2024-09-01"],
        step_up=[
            0.03,   # 3% desde 2021-09-01
            0.04,   # 4% desde 2022-09-01
            0.05,   # 5% desde 2023-09-01
            0.0525  # 5.25% desde 2024-09-01 hasta vencimiento
        ],
        outstanding=10000, calificacion="CCC-"
    )
    
    # =========================
    # BB7DD
    # =========================
    bb7dd = bond_calculator_pro(
        name="BB7DD", emisor="Provincia Buenos Aires", curr="CCL/MEP", law="NY",
        start_date=datetime(2021,6,30), end_date=datetime(2037,9,1),
        payment_frequency=6,
        amortization_dates=[
            "2024-03-01","2024-09-01",
            "2025-03-01","2025-09-01",
            "2026-03-01","2026-09-01",
            "2027-03-01","2027-09-01",
            "2028-03-01","2028-09-01",
            "2029-03-01","2029-09-01",
            "2030-03-01","2030-09-01",
            "2031-03-01","2031-09-01",
            "2032-03-01","2032-09-01",
            "2033-03-01","2033-09-01",
            "2034-03-01","2034-09-01",
            "2035-03-01","2035-09-01",
            "2036-03-01","2036-09-01",
            "2037-03-01","2037-09-01"
        ],
        amortizations=[  # ejemplo: iguales de 3.57% (28 cuotas)
            100/28.0
        ]*28,
        rate=2.5,   # arranca 2.5% anual
        price=px("BB7DD"),
        step_up_dates=["2021-09-01","2022-09-01","2023-09-01","2024-09-01"],
        step_up=[
            0.039,   # 3.9% desde 2021-09-01
            0.0525,  # 5.25% desde 2022-09-01
            0.06375, # 6.375% desde 2023-09-01
            0.06625  # 6.625% desde 2024-09-01 hasta vencimiento
        ],
        outstanding=15000, calificacion="CCC-"
    )
    
    # =========================
    # BC7DD
    # =========================
    bc7dd = bond_calculator_pro(
        name="BC7DD", emisor="Provincia Buenos Aires", curr="CCL/MEP", law="NY",
        start_date=datetime(2021,6,30), end_date=datetime(2037,9,1),
        payment_frequency=6,
        amortization_dates=[
            "2024-03-01","2024-09-01","2025-03-01","2025-09-01",
            "2026-03-01","2026-09-01","2027-03-01","2027-09-01",
            "2028-03-01","2028-09-01","2029-03-01","2029-09-01",
            "2030-03-01","2030-09-01","2031-03-01","2031-09-01",
            "2032-03-01","2032-09-01","2033-03-01","2033-09-01",
            "2034-03-01","2034-09-01","2035-03-01","2035-09-01",
            "2036-03-01","2036-09-01","2037-03-01","2037-09-01"
        ],
        amortizations=[100/28.0]*28,  # igual que AL41
        rate=2.5,
        price=px("BC7DD"),
        step_up_dates=["2021-09-01","2022-09-01","2023-09-01","2024-09-01"],
        step_up=[
            0.035,   # 3.5% desde 2021-09-01
            0.045,   # 4.5% desde 2022-09-01
            0.055,   # 5.5% desde 2023-09-01
            0.05875  # 5.875% desde 2024-09-01 hasta vencimiento
        ],
        outstanding=12000, calificacion="CCC-"
    )

    return [gd_29, gd_30, gd_35, gd_38, gd_41, gd_46,
            al_29, al_30, al_35, ae_38, al_41,
            bpb7d, bpc7d, bpd7d, ba7dd, bb7dd, bc7dd]


# =========================
# Simulador de flujos
# =========================
def build_cashflow_table(selected_bonds: list, mode: str, inputs: dict) -> pd.DataFrame:
    rows = []
    for b in selected_bonds:
        # SIN el primer flujo (precio): solo pagos
        dates = b.generate_payment_dates()[1:]
        coupons = b.coupon_payments()[1:]
        capitals = b.amortization_payments()[1:]

        if mode == "Nominal":
            nominal = float(inputs.get(b.name, 0) or 0) / 100
        else:  # Monto
            user_in = inputs.get(b.name, {})
            monto = float(user_in.get("monto", 0) or 0)
            precio_manual = user_in.get("precio", None)
            precio = precio_manual if precio_manual else b.price
            nominal = (monto / precio) if (precio and precio == precio) else 0.0

        for d, cpn, cap in zip(dates, coupons, capitals):
            rows.append({
                "Fecha": d,
                "Ticker": b.name,
                "Cupón": round(cpn * nominal, 2),
                "Capital": round(cap * nominal, 2),
                "Total": round((cpn + cap) * nominal, 2)
            })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["Fecha", "Cupón", "Capital", "Total"])

    df_total = df.groupby("Fecha", as_index=False)[["Cupón", "Capital", "Total"]].sum()
    df_total["Cupón"] = df_total["Cupón"].round(2)
    df_total["Capital"] = df_total["Capital"].round(2)
    df_total["Total"] = df_total["Total"].round(2)

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

# ------------------------
# LECAPs / BONCAPs definidos a nivel módulo
# ------------------------

LECAPS_ROWS = [
    ("S30S5","30/9/2025","30/9/2024",3.98, "Fija"),
    ("T17O5","17/10/2025","14/10/2024",3.90, "Fija"),
    ("S31O5","31/10/2025","16/12/2024",2.74, "Fija"),
    ("S10N5","10/11/2025","31/01/2025",2.2,  "Fija"),
    ("S28N5","28/11/2025","14/2/2025", 2.26, "Fija"),
    ("T15D5","15/12/2025","14/10/2024",3.89, "Fija"),
    ("S16E6","16/01/2026","18/08/2025",3.6,  "Fija"),
    ("T30E6","30/1/2026","16/12/2024", 2.65, "Fija"),
    ("T13F6","13/2/2026","29/11/2024", 2.60, "Fija"),
    ("S27F6","27/2/2026","29/8/2025",  3.95, "Fija"),
    ("S29Y6","29/5/2026","30/5/2025",  2.35, "Fija"),
    ("T30J6","30/6/2026","17/1/2025",  2.15, "Fija"),
    ("T15E7","15/1/2027","31/1/2025",  2.05, "Fija"),
    ("TTM26","16/3/2026","29/1/2025",  2.225,"Fija"),
    ("TTJ26","30/6/2026","29/1/2025",  2.19, "Fija"),
    ("TTS26","15/9/2026","29/01/2025", 2.17, "Fija"),
    ("TTD26","15/12/2026","29/01/2025",2.14, "Fija"),
]

# -------------------------------------------------
# ===== 1) Fetch robusto con cache =====
# -------------------------------------------------

@st.cache_data(ttl=60*60*12, show_spinner=False)  # cachea 12 horas
def fetch_cer_df(series_id: int = 30) -> pd.DataFrame:
    base = "https://api.bcra.gob.ar/estadisticas"
    version = "v4.0"
    url = f"{base}/{version}/monetarias/{series_id}"

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    r = session.get(
        url,
        timeout=20,
        headers={
            "Accept": "application/json",
            "User-Agent": "Mateo-Streamlit/1.0 (+contacto)"
        },
    )
    r.raise_for_status()
    js = r.json()

    # Validación básica del schema
    if "results" not in js or not js["results"]:
        raise ValueError("Respuesta sin 'results' o vacía del BCRA.")
    if "detalle" not in js["results"][0]:
        raise ValueError("No se encontró 'detalle' en results[0].")

    df = pd.DataFrame(js["results"][0]["detalle"])
    # Nos quedamos con fecha/valor y tipamos
    df = (
        df[["fecha", "valor"]]
        .assign(
            fecha=lambda d: pd.to_datetime(d["fecha"], errors="coerce"),
            valor=lambda d: pd.to_numeric(d["valor"], errors="coerce"),
        )
        .dropna(subset=["fecha", "valor"])
        .sort_values("fecha")
        .reset_index(drop=True)
    )
    return df

# ===== 2) Días hábiles: usa QuantLib si está disponible; si no, fallback Mon–Fri =====
def last_business_day_arg(lag_business_days: int = 10) -> date:
    # Intento con QuantLib (calendario Argentina Merval)
    try:
        import QuantLib as ql
        cal = ql.Argentina(ql.Argentina.Merval)

        qd = ql.Date.todaysDate() + 1  # asumiendo liquidación T+1
        # retrocedo 'lag_business_days' días hábiles
        count = 0
        while count < lag_business_days:
            qd = qd - 1
            if cal.isBusinessDay(qd):
                count += 1
        return date(qd.year(), qd.month(), qd.dayOfMonth())

    except Exception:
        # Fallback simple: cuenta solo Mon–Fri (sin feriados)
        # Si te interesa feriados reales sin QuantLib, considera 'workalendar'
        d = datetime.utcnow().date() + timedelta(days=1)
        count = 0
        while count < lag_business_days:
            d = d - timedelta(days=1)
            if d.weekday() < 5:  # 0=Mon ... 4=Fri
                count += 1
        return d

def cer_at_or_before(df: pd.DataFrame, target_day: date) -> float:
    # Filtra hasta target (incluye ese día)
    sel = df[df["fecha"].dt.date <= target_day]
    if sel.empty:
        raise ValueError("No hay datos de CER previos a la fecha objetivo.")
    return float(sel.iloc[-1]["valor"])

# =========================
# App UI
# =========================

def main():
    st.sidebar.title("Navegación")
    page = st.sidebar.radio("Elegí sección", ["Bonos HD", "Lecaps", "Otros"], index=0)

    # --- Carga de mercado + botón refrescar ---
    with st.spinner("Cargando precios de mercado..."):
        try:
            df_all, df_mep = load_market_data()
        except Exception as e:
            st.error(f"Error al cargar precios de mercado: {e}")
            df_all, df_mep = pd.DataFrame(), pd.DataFrame()

    if st.sidebar.button("🔄 Actualizar ahora"):
        load_market_data.clear()
        with st.spinner("Actualizando..."):
            try:
                df_all, df_mep = load_market_data()
                st.sidebar.success("Precios actualizados.")
            except Exception as e:
                st.sidebar.error(f"Error al actualizar: {e}")
                df_all, df_mep = pd.DataFrame(), pd.DataFrame()

    df_all_norm = normalize_market_df(df_all)

    # --- Construcción de universos ---
    try:
        ons_bonds = load_bcp_from_excel(df_all, adj=1.005, price_col_prefer="px_ask")
    except Exception as e:
        st.warning(f"No se pudo cargar el listado de ONs: {e}")
        ons_bonds = []
    manual_bonds = manual_bonds_factory(df_all)
    all_bonds = ons_bonds + manual_bonds
    name_to_bond = {b.name: b for b in all_bonds}

    if page == "Bonos HD":
        st.title("Bonos HD")
        st.caption("Explorar métricas de Bonos HD")

        # =========================
        # 1) TABLA DE MÉTRICAS + FILTROS
        # =========================
        st.subheader("Métricas")
        df_full = metrics_bcp(all_bonds)
        
        # Filtros
        colf1, colf2, colf3, colf4 = st.columns(4)
        
        emisores = sorted([e for e in df_full["Emisor"].dropna().unique()])
        monedas  = sorted([m for m in df_full["Moneda de Pago"].dropna().unique()])
        leyes    = sorted([l for l in df_full["Ley"].dropna().unique()])
        tickers  = sorted([t for t in df_full["Ticker"].dropna().unique()])
        
        with colf1:
            all_emisores = st.checkbox("Todos los emisores", value=True)
            f_emisor = st.multiselect("Filtrar Emisor", emisores, default=emisores if all_emisores else [])
            if all_emisores:
                f_emisor = emisores
        
        with colf2:
            all_monedas = st.checkbox("Todas las monedas", value=True)
            f_moneda = st.multiselect("Filtrar Moneda de Pago", monedas, default=monedas if all_monedas else [])
            if all_monedas:
                f_moneda = monedas
        
        with colf3:
            all_leyes = st.checkbox("Todas las leyes", value=True)
            f_ley = st.multiselect("Filtrar Ley", leyes, default=leyes if all_leyes else [])
            if all_leyes:
                f_ley = leyes
        
        with colf4:
            all_tickers = st.checkbox("Todos los tickers", value=True)
            f_ticker = st.multiselect("Filtrar Ticker", tickers, default=tickers if all_tickers else [])
            if all_tickers:
                f_ticker = tickers
        
        # Aplicar filtros
        mask = (
            df_full["Emisor"].isin(f_emisor)
            & df_full["Moneda de Pago"].isin(f_moneda)
            & df_full["Ley"].isin(f_ley)
            & df_full["Ticker"].isin(f_ticker)
        )
        df_filtered = df_full.loc[mask].reset_index(drop=True)
        
        # Mostrar DataFrame directo en Streamlit
        st.dataframe(
            df_filtered.style.format({
                "Precio": "{:.1f}",
                "TIR": "{:.1f}",
                "TNA SA": "{:.1f}",
                "Duration": "{:.1f}",
                "Modified Duration": "{:.1f}",
                "Convexidad": "{:.1f}",
                "Paridad": "{:.1f}",
                "Current Yield": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True
        )
        
        st.divider()

        # =========================
        # 2) SIMULADOR DE FLUJOS
        # =========================
        st.subheader("Simulador de Flujos")

        colA, colB = st.columns([1, 2])
        with colA:
            sel_bonds = st.multiselect(
                "Seleccioná bonos",
                options=sorted(name_to_bond.keys()),
                default=[]
            )
            mode = st.radio("Modo de entrada", ["Nominal", "Monto"], horizontal=True, index=0)

        with colB:
            inputs = {}
            if sel_bonds:
                st.write("Parámetros por bono:")
                for n in sel_bonds:
                    if mode == "Nominal":
                        vn = st.number_input(
                            f"VN de {n}", min_value=0.0, step=100.0, value=0.0, key=f"vn_{n}"
                        )
                        inputs[n] = vn
                    else:  # Monto
                        monto = st.number_input(
                            f"Monto (USD) para {n}", min_value=0.0, step=100.0, value=0.0, key=f"monto_{n}"
                        )
                        precio_manual = st.number_input(
                            f"Precio manual (opcional) para {n}", min_value=0.0, step=0.1, value=0.0, key=f"precio_{n}"
                        )
                        inputs[n] = {"monto": monto, "precio": precio_manual if precio_manual > 0 else None}

        if sel_bonds:
            selected_objs = [name_to_bond[n] for n in sel_bonds]
            df_cf = build_cashflow_table(selected_objs, mode, inputs)  # ← devuelve Fecha, Cupón, Capital, Total

            st.markdown("**Flujo consolidado por fecha (USD):**")
            st.dataframe(
                df_cf,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Cupón":  st.column_config.NumberColumn(format="%.2f"),
                    "Capital": st.column_config.NumberColumn(format="%.2f"),
                    "Total":  st.column_config.NumberColumn(format="%.2f"),
                },
            )
        else:
            st.info("Seleccioná al menos un bono para ver flujos.")

        st.divider()

        # =========================
        # 3) Calculadora de Métricas
        # =========================
        def compute_metrics_with_price(b: bond_calculator_pro, price_override: float | None = None, settlement=None) -> dict:
            # 1) crear un clon con el precio pedido
            price = float(price_override) if price_override and price_override > 0 else b.price
            bb = clone_with_price(b, price)
        
            # 2) (opcional) invalidar cachés internos si existen
            if hasattr(bb, "_yield_cache"): bb._yield_cache.clear()
            if hasattr(bb, "_duration_cache"): bb._duration_cache.clear()
        
            row = {
                "Ticker": bb.name,
                "Precio": round(bb.price, 1),
                "TIR": bb.xirr(settlement),
                "TNA SA": bb.tna_180(settlement),
                "Duration": bb.duration(settlement),
                "Modified Duration": bb.modified_duration(settlement),
                "Convexidad": bb.convexity(settlement),
                "Paridad": bb.parity(settlement),
                "Current Yield": bb.current_yield(settlement),
            }
            for k in ("TIR","TNA SA","Duration","Modified Duration","Convexidad","Paridad","Current Yield"):
                row[k] = round(pd.to_numeric(row[k], errors="coerce"), 1)
            return row

        st.subheader("Comparador de Métricas (3 bonos)")
        col1, col2, col3 = st.columns(3)
        choices = sorted(name_to_bond.keys())

        with col1:
            b1_name = st.selectbox("Bono 1", choices, index=0, key="cmp_b1")
            p1 = st.number_input("Precio manual 1 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p1")
        with col2:
            b2_name = st.selectbox("Bono 2", choices, index=1, key="cmp_b2")
            p2 = st.number_input("Precio manual 2 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p2")
        with col3:
            b3_name = st.selectbox("Bono 3", choices, index=2, key="cmp_b3")
            p3 = st.number_input("Precio manual 3 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p3")

        if st.button("Calcular comparativa"):
            rows = []
            for nm, pv in [(b1_name, p1), (b2_name, p2), (b3_name, p3)]:
                if not nm:
                    continue
                b = name_to_bond[nm]
                price_override = pv if pv and pv > 0 else None
                rows.append(compute_metrics_with_price(b, price_override))
            if rows:
                df_cmp = pd.DataFrame(rows, columns=[
                    "Ticker","Precio","TIR","TNA SA","Duration","Modified Duration","Convexidad","Paridad","Current Yield"
                ])
                st.dataframe(
                    df_cmp.style.format({
                        "Precio": "{:.1f}",
                        "TIR": "{:.1f}",
                        "TNA SA": "{:.1f}",
                        "Duration": "{:.1f}",
                        "Modified Duration": "{:.1f}",
                        "Convexidad": "{:.1f}",
                        "Paridad": "{:.1f}",
                        "Current Yield": "{:.1f}",
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("Elegí al menos un bono.")
        
        # =========================
        # 4) Curvas comparadas por Emisor (TIR vs Modified Duration)
        # =========================
        st.subheader("Curvas comparadas por Emisor (TIR vs Modified Duration)")

        # Parto de las métricas ya calculadas (o recalculo por seguridad)
        df_metrics = metrics_bcp(all_bonds).copy()

        # Emitores disponibles
        emisores_all = sorted([e for e in df_metrics["Emisor"].dropna().unique()])

        colc1, colc2, colc3 = st.columns([1,1,2])
        with colc1:
            em1 = st.selectbox("Emisor A", emisores_all, index=0, key="curve_em1")
        with colc2:
            # por defecto un emisor diferente a em1 si existe
            idx_default = 1 if len(emisores_all) > 1 else 0
            em2 = st.selectbox("Emisor B", emisores_all, index=idx_default, key="curve_em2")
        with colc3:
            st.caption("Gráfico: eje X = Modified Duration | eje Y = TIR (e.a. %)")

        # Filtro por los dos emisores seleccionados
        emisores_sel = [em1, em2] if em1 != em2 else [em1]
        df_curves = df_metrics[df_metrics["Emisor"].isin(emisores_sel)].copy()

        # Asegurar numéricos y 1 decimal para la tabla
        for c in ["TIR", "Modified Duration", "Precio", "TNA SA", "Convexidad", "Paridad", "Current Yield"]:
            if c in df_curves.columns:
                df_curves[c] = pd.to_numeric(df_curves[c], errors="coerce")

        # Scatter interactivo
        if not df_curves.empty:
            fig = px.scatter(
                df_curves,
                x="Modified Duration",
                y="TIR",
                color="Emisor",
                symbol="Emisor",
                hover_name="Ticker",
                hover_data={
                    "Emisor": True,
                    "Ticker": False,
                    "Ley": True,
                    "Moneda de Pago": True,
                    "Precio": ":.1f",
                    "TIR": ":.1f",
                    "TNA SA": ":.1f",
                    "Modified Duration": ":.1f",
                    "Convexidad": ":.1f",
                    "Paridad": ":.1f",
                    "Current Yield": ":.1f",
                },
                size_max=12,
            )
            fig.update_traces(marker=dict(size=12, line=dict(width=1)))
            fig.update_layout(
                xaxis_title="Modified Duration (años)",
                yaxis_title="TIR (%)",
                legend_title="Emisor",
                height=480,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Tabla debajo (1 decimal, sin índice)
            st.markdown("**Bonos incluidos en las curvas:**")
            cols_show = [
                "Ticker","Emisor","Ley","Moneda de Pago","Precio",
                "TIR","TNA SA","Modified Duration","Convexidad","Paridad","Current Yield",
                "Próxima Fecha de Pago","Fecha de Vencimiento"
            ]
            cols_show = [c for c in cols_show if c in df_curves.columns]
            st.dataframe(
                df_curves[cols_show].style.format({
                    "Precio": "{:.1f}",
                    "TIR": "{:.1f}",
                    "TNA SA": "{:.1f}",
                    "Modified Duration": "{:.1f}",
                    "Convexidad": "{:.1f}",
                    "Paridad": "{:.1f}",
                    "Current Yield": "{:.1f}",
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No hay bonos para los emisores seleccionados.")

    elif page == "Lecaps":
        st.title("LECAPs / BONCAPs / TAMAR")
    
        # Normalizar y armar tabla (precios ASK*1.005 ya aplicados en build_lecaps_metrics si hiciste el ajuste anterior)
        df_all_norm = normalize_market_df(df_all)
        df_lecaps = build_lecaps_metrics(LECAPS_ROWS, df_all_norm)
    
        st.subheader("Métricas de LECAPs/BONCAPs")
        st.dataframe(df_lecaps, use_container_width=True, hide_index=True)
    
        # ---------- Objetos para cálculos (solo LECAPs) ----------
        le_map = build_lecaps_objects(LECAPS_ROWS, df_all_norm)
    
        st.divider()
        st.subheader("Precio ↔ Rendimiento (LECAPs/BONCAPs)")
    
        tab_prc, tab_yld = st.tabs(["Precio → Rendimiento", "Rendimiento → Precio"])
    
        with tab_prc:
            if not le_map:
                st.info("No se pudieron construir objetos LECAP. Verificá precios de mercado.")
            else:
                tickers = sorted(le_map.keys())
                bname = st.selectbox("Elegí LECAP/BONCAP", tickers, key="lx_px2y")
                prc_in = st.number_input("Precio (clean) → TIR e.a. (%)", min_value=0.0, step=0.1, value=0.0, key="lx_px")
                if st.button("Calcular TIR", key="btn_lx_px2y"):
                    b = le_map[bname]
                    y = b.yield_from_price(prc_in)
                    if np.isnan(y):
                        st.error("No se pudo calcular la TIR con ese precio.")
                    else:
                        st.success(f"TIR efectiva anual: **{y:.2f}%**")
    
                        # Métricas a ese precio (sin mutar el original)
                        old = b.price
                        try:
                            b.price = prc_in
                            df_one = pd.DataFrame([{
                                "Ticker": b.name,
                                "Precio": prc_in,
                                "TIR": b.xirr(),
                                "TNA 30": b.tna30(),
                                "Duration": b.duration(),
                                "Modified Duration": b.modified_duration(),
                                "Retorno Directo": b.direct_return(),
                            }])
                        finally:
                            b.price = old
    
                        for c in ["Precio","TIR","TNA 30","Duration","Modified Duration","Retorno Directo"]:
                            df_one[c] = pd.to_numeric(df_one[c], errors="coerce")
                        df_one["Precio"] = df_one["Precio"].round(2)
                        for c in ["TIR","TNA 30","Duration","Modified Duration","Retorno Directo"]:
                            df_one[c] = df_one[c].round(2)
                        st.dataframe(df_one, use_container_width=True, hide_index=True)
    
        with tab_yld:
            if not le_map:
                st.info("No se pudieron construir objetos LECAP. Verificá precios de mercado.")
            else:
                tickers2 = sorted(le_map.keys())
                bname2 = st.selectbox("Elegí LECAP/BONCAP", tickers2, key="lx_y2px")
                yld_in = st.number_input("TIR e.a. (%) → Precio (clean)", min_value=-99.0, step=0.1, value=0.0, key="lx_y")
                if st.button("Calcular Precio", key="btn_lx_y2px"):
                    b2 = le_map[bname2]
                    p = b2.price_from_irr(yld_in)
                    if np.isnan(p):
                        st.error("No se pudo calcular el precio con esa TIR.")
                    else:
                        st.success(f"Precio clean: **{p:.2f}**")
                        # Chequeo TIR con ese precio
                        tir_check = b2.yield_from_price(p)
                        st.caption(f"Chequeo: TIR con ese precio = **{tir_check:.2f}%**")
    
        # ---------- Curva excluyendo TTM, TTJ, TTS, TTD ----------
        st.divider()
        st.subheader("Curva Tasa Fija")

        if df_lecaps.empty:
            st.info("No hay datos de LECAPs/BONCAPs para graficar.")
        else:
            excl = {"TTM26", "TTJ26", "TTS26", "TTD26"}
            df_curve = df_lecaps.copy()
            df_curve = df_curve[~df_curve["Ticker"].isin(excl)].copy()

            # Asegurar numéricos
            for c in ["TNA 30", "Modified Duration", "Precio", "Duration", "Retorno Directo", "Rendimiento (TIR EA)", "TEM"]:
                if c in df_curve.columns:
                    df_curve[c] = pd.to_numeric(df_curve[c], errors="coerce")

            # Vamos a graficar directamente TNA 30 (eje Y) vs Modified Duration (eje X)
            df_plot = df_curve.dropna(subset=["TNA 30", "Modified Duration"]).copy()

            if df_plot["Modified Duration"].gt(0).sum() == 0:
                st.info("No hay Modified Duration > 0 para ajustar una curva logarítmica.")
            else:
                # --- Scatter con etiquetas por ticker ---
                fig = px.scatter(
                    df_plot,
                    x="Modified Duration",
                    y="TNA 30",
                    color="Tipo" if "Tipo" in df_plot.columns else None,
                    hover_name="Ticker",
                    text="Ticker",
                    hover_data={
                        "Ticker": False,
                        "Tipo": True if "Tipo" in df_plot.columns else False,
                        "Vencimiento": True if "Vencimiento" in df_plot.columns else False,
                        "Precio": ":.2f" if "Precio" in df_plot.columns else False,
                        "Rendimiento (TIR EA)": ":.2f" if "Rendimiento (TIR EA)" in df_plot.columns else False,
                        "TEM": ":.2f" if "TEM" in df_plot.columns else False,
                        "Duration": ":.2f" if "Duration" in df_plot.columns else False,
                        "Modified Duration": ":.2f",
                        "Retorno Directo": ":.2f" if "Retorno Directo" in df_plot.columns else False,
                        "TNA 30": ":.2f",
                    },
                    size_max=12,
                )
                fig.update_traces(
                    marker=dict(size=12, line=dict(width=1)),
                    textposition="top center",
                    textfont=dict(size=10)
                )

                # --- Ajuste logarítmico global: TNA30 = a + b * ln(MD) ---
                df_fit = df_plot[["Modified Duration", "TNA 30"]].dropna()
                df_fit = df_fit[df_fit["Modified Duration"] > 0]   # ln(x) requiere x>0

                x = df_fit["Modified Duration"].to_numpy(dtype=float)
                y = df_fit["TNA 30"].to_numpy(dtype=float)
                Xlog = np.log(x)

                # Coeficientes (y = a + b*ln(x))
                bcoef, acoef = np.polyfit(Xlog, y, 1)

                # Curva lisa para dibujar
                x_line = np.linspace(x.min(), x.max(), 200)
                y_line = acoef + bcoef * np.log(x_line)

                fig.add_trace(
                    go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name="Ajuste log: TNA30 = a + b·ln(MD)"
                    )
                )

                # R^2 para referencia
                y_hat = acoef + bcoef * Xlog
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

                fig.update_layout(
                    xaxis_title="Modified Duration (años)",
                    yaxis_title="TNA 30 (%)",
                    legend_title="Tipo" if "Tipo" in df_plot.columns else None,
                    height=480,
                    margin=dict(l=10, r=10, t=10, b=10),
                    annotations=[
                        dict(
                            x=0.99, y=0.02, xref="paper", yref="paper",
                            xanchor="right", showarrow=False,
                            text=f"a={acoef:.2f}, b={bcoef:.2f}, R²={r2:.2f}"
                        )
                    ],
                )
                st.plotly_chart(fig, use_container_width=True)
    
        # ---------- TC implícito MEP->LECAP/BONCAP + Bandas ----------
        st.divider()
        st.subheader("TC implícito al vencimiento (MEP→LECAP/BONCAP) con bandas")
    
        # Intento obtener un valor por defecto de MEP a partir de AL30
        try:
            if isinstance(df_mep, pd.DataFrame) and not df_mep.empty:
                mep_default = float(
                    df_mep.loc[df_mep["ticker"].str.upper() == "AL30", "close"].iloc[0]
                )
            else:
                mep_default = 1000.0
        except Exception:
            mep_default = 1000.0

        # El usuario lo puede modificar
        mep_rate = st.number_input(
            "MEP actual (ARS por USD)", 
            min_value=0.0, 
            step=1.0, 
            value=mep_default, 
            key="mep_rate_input"
        )
    
        # Selección de tickers a mostrar
        tickers_fx = sorted(le_map.keys()) if le_map else []
        sel_fx = st.multiselect("Elegí LECAPs/BONCAPs para el gráfico de TC implícito", tickers_fx, default=tickers_fx)
    
        if not le_map or not sel_fx:
            st.info("Seleccioná al menos una LECAP/BONCAP.")
        else:
            # Pago final por 100 VN en LECAP/BONCAP
            def final_payment(obj: lecaps) -> float:
                # replica lógica: 100 * (1+TEM)^(meses_30/360)
                capital = 100.0
                months = obj._months_30_360()
                return capital * ((1.0 + obj.tem) ** months)
    
            # Bandas: parten 2025-04-07 en 1400/1000 y crecen 1% mensual prorrateado por días hábiles (Merval).
            base_date = datetime(2025, 4, 7)
            cal = ql.Argentina(ql.Argentina.Merval)
            per_bd_factor = 1.01 ** (1.0 / 21.0)  - 1# 1% por ~21 hábiles/mes
    
            rows_fx = []
            for tkr in sel_fx:
                obj = le_map.get(tkr)
                if obj is None:
                    continue
                price = float(obj.price) if np.isfinite(obj.price) else np.nan
                if not np.isfinite(price) or price <= 0:
                    continue
    
                # TC implícito: (MEP * pago_final) / precio
                pf = final_payment(obj)
                tc_impl = mep_rate * (pf / price)
    
                # bdays desde 2025-04-07 hasta el vencimiento
                dt_mty = obj.end_date
                qd0 = ql.Date(base_date.day, base_date.month, base_date.year)
                qd1 = ql.Date(dt_mty.day, dt_mty.month, dt_mty.year)
                bdays = cal.businessDaysBetween(qd0, qd1, False, False)  # sin contar extremos
    
                band_upper = 1400.0 * ((1 + per_bd_factor) ** bdays)
                band_lower = 1000.0 * ((1 - per_bd_factor) ** bdays)
    
                rows_fx.append({
                    "Ticker": tkr,
                    "Vencimiento": dt_mty.date(),
                    "Precio": price,
                    "Pago final por 100": pf,
                    "TC implícito": tc_impl,
                    "Banda sup": band_upper,
                    "Banda inf": band_lower,
                })
    
            df_fx = pd.DataFrame(rows_fx)
            if df_fx.empty:
                st.info("No hay datos suficientes para calcular el TC implícito.")
            else:
                # Gráfico: puntos por Ticker (TC implícito) + líneas de bandas vs. Vencimiento
                df_fx = df_fx.sort_values("Vencimiento")
    
                # Serie de fechas y bandas
                x_dates = pd.to_datetime(df_fx["Vencimiento"]).to_list()
                y_sup = df_fx["Banda sup"].to_list()
                y_inf = df_fx["Banda inf"].to_list()
    
                fig_fx = go.Figure()
                # Bandas
                fig_fx.add_trace(go.Scatter(
                    x=x_dates, y=y_sup, mode="lines", name="Banda superior (1%/mes háb.)"
                ))
                fig_fx.add_trace(go.Scatter(
                    x=x_dates, y=y_inf, mode="lines", name="Banda inferior (1%/mes háb.)"
                ))
                # Puntos TC implícito
                fig_fx.add_trace(go.Scatter(
                    x=x_dates,
                    y=df_fx["TC implícito"],
                    mode="markers+text",
                    name="TC implícito (MEP→LECAP/BONCAP)",
                    text=df_fx["Ticker"],
                    textposition="top center",
                    hovertext=[
                        f"{r['Ticker']} | Vto: {r['Vencimiento']}<br>"
                        f"Precio: {r['Precio']:.2f} | Pago100: {r['Pago final por 100']:.2f}<br>"
                        f"TC implícito: {r['TC implícito']:.2f}"
                        for _, r in df_fx.iterrows()
                    ],
                    hoverinfo="text",
                    marker=dict(size=12, line=dict(width=1))
                ))
                fig_fx.update_layout(
                    xaxis_title="Fecha de vencimiento",
                    yaxis_title="ARS por USD",
                    legend_title="Series",
                    height=480,
                    margin=dict(l=10, r=10, t=10, b=10),
                )
                st.plotly_chart(fig_fx, use_container_width=True)
 
    else:
        # ===== 3) UI Streamlit =====
        st.title("CER (BCRA) con t - 10 días hábiles")
        
        with st.status("Consultando API del BCRA…", expanded=False):
            df_cer = fetch_cer_df(30)
        
        st.success(f"Datos CER cargados. Última fecha disponible: {df_cer['fecha'].max().date()}")
        
        lag = st.number_input("Días hábiles hacia atrás", min_value=1, max_value=60, value=10, step=1)
        target = last_business_day_arg(int(lag))
        
        try:
            cer_final = cer_at_or_before(df_cer, target)
            st.metric(label=f"CER a t-{lag} hábiles (target: {target})", value=f"{cer_final:,.2f}")
        except Exception as e:
            st.error(f"No se pudo calcular CER a t-{lag} hábiles: {e}")
        
        # Muestra rápida del DF
        st.dataframe(df_cer.tail(20), use_container_width=True)
        
        # Descarga opcional
        st.download_button(
            "Descargar CER (CSV)",
            data=df_cer.to_csv(index=False).encode("utf-8"),
            file_name="cer_bcra.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
