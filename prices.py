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
        residual, current = [], 100.0
        for d in self.generate_payment_dates():
            if d in self.amortization_dates:
                idx = self.amortization_dates.index(d)
                current -= self.amortizations[idx]
            residual.append(current)
        return residual

    def accrued_interest(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        ref_date = self._as_dt(ref_date)

        dates_str = self.generate_payment_dates()
        all_dt = [datetime.strptime(s, "%Y-%m-%d") for s in dates_str]
        coup_dt = all_dt[1:]  # exclude t0 (settlement)

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
        period_start = self._as_dt(self.start_date) if idx == 0 else coup_dt[idx - 1]

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
        cpns, residuals = [], self.residual_value()
        for i, _ in enumerate(self.generate_payment_dates()):
            cpns.append(0.0 if i == 0 else (self.rate / self.frequency) * residuals[i - 1])
        return cpns

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

def parse


