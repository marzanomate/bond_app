# app.py
import re
import io
import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# =====================================
# Config
# =====================================
st.set_page_config(page_title="Calculadora ONs", layout="wide")

EXCEL_URL_DEFAULT = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"
URL_BONDS = "https://data912.com/live/arg_bonds"
URL_NOTES = "https://data912.com/live/arg_notes"
URL_CORPS = "https://data912.com/live/arg_corp"

# =====================================
# Clase de ON (sin 'frequency' explícito)
# =====================================
class ons_pro:
    def __init__(self, name, empresa, curr, law, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price):
        self.name = name
        self.empresa = empresa
        self.curr = curr
        self.law = law
        self.start_date = start_date
        self.end_date = end_date
        self.payment_frequency = int(payment_frequency)  # meses
        if self.payment_frequency <= 0:
            raise ValueError(f"{name}: payment_frequency debe ser > 0")
        self.amortization_dates = amortization_dates     # YYYY-MM-DD (str)
        self.amortizations = amortizations               # por 100 nominal
        self.rate = float(rate) / 100.0                  # % nominal anual -> decimal
        self.price = float(price)                        # clean por 100 nominal

    def _freq(self):
        return max(1, int(round(12 / self.payment_frequency)))  # cupones/año

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
        settlement = datetime.today() + timedelta(days=1)
        back = []
        cur = self._as_dt(self.end_date)
        start = self._as_dt(self.start_date)
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
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        return [self.outstanding_on(d) for d in dates]

    def accrued_interest(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        ref_date = self._as_dt(ref_date)
        all_dt = [self._as_dt(s) for s in self.generate_payment_dates()]
        coup_dt = all_dt[1:]
        if not coup_dt:
            return 0.0
        next_coupon = next((d for d in coup_dt if d > ref_date), None)
        if next_coupon is None:
            return 0.0
        idx = coup_dt.index(next_coupon)
        period_start = (max(self._as_dt(self.start_date),
                            next_coupon - relativedelta(months=self.payment_frequency))
                        if idx == 0 else coup_dt[idx-1])
        base = self.outstanding_on(period_start)
        full_coupon = (self.rate / self._freq()) * base
        total_days = max(1, (next_coupon - period_start).days)
        accrued_days = max(0, min((ref_date - period_start).days, total_days))
        return full_coupon * (accrued_days / total_days)

    def parity(self, ref_date=None):
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)
        vt = self.outstanding_on(ref_date) + self.accrued_interest(ref_date)
        return float('nan') if vt == 0 else round(self.price / vt * 100.0, 2)

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

    def cash_flow(self):
        cfs = []
        dates = self.generate_payment_dates()
        caps = self.amortization_payments()
        cpns = self.coupon_payments()
        for i, _ in enumerate(dates):
            cfs.append(-self.price if i == 0 else caps[i] + cpns[i])
        return cfs

    def xnpv(self, rate_custom=0.08):
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]
        cfs = self.cash_flow()
        d0 = datetime.today() + timedelta(days=1)
        return sum(cf / (1.0 + rate_custom) ** ((dt - d0).days / 365.0)
                   for cf, dt in zip(cfs, dates))

    def xirr(self):
        # bisección con auto-bracketing (evita SciPy)
        def f(r): return self.xnpv(rate_custom=r)
        lows  = [-0.99, -0.50, -0.20, -0.10, 0.0]
        highs = [0.10, 0.20, 0.50, 1.0, 2.0, 5.0, 10.0]
        for lo in lows:
            flo = f(lo)
            if np.isnan(flo):
                continue
            for hi in highs:
                if hi <= lo:
                    continue
                fhi = f(hi)
                if np.isnan(fhi):
                    continue
                if flo == 0:
                    return round(lo * 100.0, 2)
                if fhi == 0:
                    return round(hi * 100.0, 2)
                if flo * fhi < 0:
                    a, b = lo, hi
                    fa, fb = flo, fhi
                    for _ in range(120):
                        m = 0.5 * (a + b)
                        fm = f(m)
                        if abs(fm) < 1e-12 or (b - a) < 1e-12:
                            return round(m * 100.0, 2)
                        if fa * fm <= 0:
                            b, fb = m, fm
                        else:
                            a, fa = m, fm
                    return round(0.5 * (a + b) * 100.0, 2)
        return float('nan')

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

# =====================================
# Parseo y precios
# =====================================
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

def parse_amorts(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    return [parse_float_cell(p) for p in str(cell).split(";")]

@st.cache_data(show_spinner=False)
def fetch_json(url):
    r = requests.get(url, timeout=25)
    r.raise_for_status()
    return r.json()

def to_df(payload):
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    return pd.json_normalize(payload)

def harmonize_prices(df):
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

@st.cache_data(show_spinner=False)
def build_df_all():
    bonds = to_df(fetch_json(URL_BONDS))
    notes = to_df(fetch_json(URL_NOTES))
    corps = to_df(fetch_json(URL_CORPS))
    dfs = []
    for d in [bonds, notes, corps]:
        if not d.empty:
            dfs.append(harmonize_prices(d))
    if not dfs:
        return pd.DataFrame(columns=["symbol","px_bid","px_ask"])
    df_all = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates(subset=["symbol"])
    for c in ["px_bid","px_ask"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    return df_all

def get_price_for_symbol(df_all, symbol, prefer="px_ask"):
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
# Loader Excel -> objetos + métricas
# =====================================
def bond_fundamentals_ons(bond_objs):
    rows = []
    for b in bond_objs:
        try:
            rows.append([
                b.name, b.empresa, b.curr, b.law, b.rate * 100, b.price,
                b.xirr(), b.tna_180(), b.duration(), b.modified_duration(),
                b.convexity(), b.current_yield(), b.parity()
            ])
        except Exception as e:
            rows.append([b.name, b.empresa, b.curr, b.law, b.price, np.nan,
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
            st.warning(f"⚠️ Error en {b.name}: {e}")
    cols = ["Ticker","Empresa","Moneda de Pago","Ley","Cupón","Precio",
            "Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]
    df = pd.DataFrame(rows, columns=cols)
    # redondeos
    for c in ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Cupón"] = df["Cupón"].round(4)
    df["Precio"] = df["Precio"].round(2)
    for c in ["Yield","TNA_180","Current Yield","Paridad (%)"]:
        df[c] = df[c].round(2)
    for c in ["Dur","MD","Conv"]:
        df[c] = df[c].round(2)
    return df

@st.cache_data(show_spinner=False)
def load_ons_from_excel(path_or_bytes, df_all, price_col_prefer="px_ask"):
    required = ["name","empresa","curr","law","start_date","end_date",
                "payment_frequency","amortization_dates","amortizations","rate"]
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

def bond_flows_frame(b):
    dates = b.generate_payment_dates()
    res   = b.residual_value()
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
    

excel_source = EXCEL_URL_DEFAULT

# =====================================
# UI
# =====================================
st.title("📈 Obligaciones Negociables")

# Botón actualizar precios
col_header = st.columns([1, 1, 6])
with col_header[0]:
    if st.button("🔄 Actualizar precios", type="primary", help="Refresca precios de data912", key="refresh_prices"):
        st.cache_data.clear()
        st.rerun()


# Carga de precios y excel
with st.spinner("Cargando precios y Excel..."):
    df_all = build_df_all()
    if df_all.empty:
        st.error("No hay precios disponibles (data912).")
        st.stop()
    try:
        excel_bytes = io.BytesIO(requests.get(excel_source, timeout=25).content)
    except Exception as e:
        st.error(f"No pude descargar el Excel desde la URL: {e}")
        st.stop()
    bonds = load_ons_from_excel(excel_bytes, df_all, price_col_prefer="px_ask")
    if not bonds:
        st.error("El Excel no produjo bonos válidos.")
        st.stop()
    df_metrics = bond_fundamentals_ons(bonds)

# Filtros
fc = st.columns(3)
with fc[0]:
    emp_opts = sorted(df_metrics["Empresa"].dropna().unique().tolist())
    sel_emp = st.multiselect("Empresa", emp_opts, default=emp_opts, key="filter_emp")
with fc[1]:
    mon_opts = sorted(df_metrics["Moneda de Pago"].dropna().unique().tolist())
    sel_mon = st.multiselect("Moneda de Pago", mon_opts, default=mon_opts, key="filter_mon")
with fc[2]:
    ley_opts = sorted(df_metrics["Ley"].dropna().unique().tolist())
    sel_ley = st.multiselect("Ley", ley_opts, default=ley_opts, key="filter_ley")

mask = (
    df_metrics["Empresa"].isin(sel_emp) &
    df_metrics["Moneda de Pago"].isin(sel_mon) &
    df_metrics["Ley"].isin(sel_ley)
)
df_view = df_metrics.loc[mask].reset_index(drop=True)
st.dataframe(df_view, use_container_width=True, height=420)

# Descargar CSV filtrado
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
        if pick in bmap:
            b = bmap[pick]
            st.write(f"**{pick}** — Empresa: {b.empresa} · Moneda: {b.curr} · Ley: {b.law} · Cupón: {round(b.rate*100,4)}% · Precio (px_ask): {b.price:.2f}")

            df_flows = bond_flows_frame(b)

            # Escalado (excluyendo fila 0)
            if mode == "Por nominales (VN)":
                scale = (vn or 0.0) / 100.0
            else:
                # escala = monto / precio_manual (número de "bloques" de 100 nominales)
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

# Fila de etiquetas para alinear visualmente
with colM1: st.markdown("**Ticker**")
with colM2: st.markdown("**Precio manual**")
with colM3: st.markdown("** **")  # espacio (non-breaking) para ocupar el alto de la etiqueta
with colM4: st.markdown("**Resultado**")

# Widgets con labels colapsadas para mantener la línea
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
    # Botón rojo (primary) alineado en la misma fila
    go_btn = st.button("Calcular métricas", key="calc_metrics_btn", type="primary", use_container_width=True)

with colM4:
    if go_btn and tick2 and tick2 != "(ninguno)":
        bmap = {b.name: b for b in bonds}
        if tick2 in bmap:
            b0 = bmap[tick2]
            # clon con precio manual
            b = ons_pro(
                name=b0.name, empresa=b0.empresa, curr=b0.curr, law=b0.law,
                start_date=b0.start_date, end_date=b0.end_date, payment_frequency=b0.payment_frequency,
                amortization_dates=b0.amortization_dates, amortizations=b0.amortizations,
                rate=b0.rate*100.0,  # __init__ recibe % nominal anual
                price=pman
            )
            df_one = bond_fundamentals_ons([b])
            st.dataframe(df_one, use_container_width=True, height=120)
        else:
            st.warning(f"No encontré el bono {tick2} para el cálculo manual.")

