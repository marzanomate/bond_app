from __future__ import annotations
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
from dateutil.relativedelta import relativedelta


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
