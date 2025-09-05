# app.py  — Calculadora ONs con optimizaciones de performance
# =====================================================================================
# Todas las líneas tienen comentarios explicando qué hace cada instrucción.
# =====================================================================================

import re  # Expresiones regulares para validar/parsear fechas/strings.
import io  # BytesIO para manejar el Excel en memoria.
import numpy as np  # Cálculo numérico y arrays vectorizados.
import pandas as pd  # Manipulación de DataFrames.
import requests  # HTTP para traer JSON/Excel remotos.
import streamlit as st  # Framework de UI.
from datetime import datetime, timedelta  # Manejo de fechas absolutas y offsets.
from dateutil.relativedelta import relativedelta  # Restar/Agregar meses exactos en calendarios financieros.
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor  # Paralelismo (I/O y CPU).

# =====================================
# Config
# =====================================
st.set_page_config(page_title="Calculadora ONs", layout="wide")  # Título y layout ancho.

EXCEL_URL_DEFAULT = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"  # URL del Excel.
URL_BONDS = "https://data912.com/live/arg_bonds"   # Endpoint 1: soberanos/bonos.
URL_NOTES = "https://data912.com/live/arg_notes"   # Endpoint 2: letras/notas.
URL_CORPS = "https://data912.com/live/arg_corp"    # Endpoint 3: corporativos.

# =====================================
# Utilidades cacheadas (network)
# =====================================

@st.cache_data(ttl=3600, show_spinner=False)  # Cachea por una hora el Excel (reduce cold-start).
def fetch_excel_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=25)  # Descarga el Excel remoto.
    r.raise_for_status()  # Lanza excepción si el status no es 2xx.
    return r.content  # Devuelve bytes crudos (listos para BytesIO).

@st.cache_data(ttl=90, show_spinner=False)  # Cachea 90s la descarga JSON; botón "Actualizar" limpia todo.
def fetch_json(url: str):
    r = requests.get(url, timeout=25)  # GET con timeout.
    r.raise_for_status()  # Excepción si falla.
    return r.json()  # Devuelve payload JSON (dict/list).

# =====================================
# Normalización de payloads de precios
# =====================================

def to_df(payload):
    # Convierte JSON a DataFrame, buscando claves comunes (data/results/items/bonds/notes).
    if isinstance(payload, dict):  # Si es dict, intenta hallar la lista interna.
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):  # Si hay lista, úsala.
                payload = payload[key]
                break
    return pd.json_normalize(payload)  # Aplana estructuras anidadas a columnas.

def harmonize_prices(df):
    # Renombra columnas heterogéneas a un esquema común: symbol, px_bid, px_ask.
    rename_map = {}  # Mapa de renombres.
    cols = {c.lower(): c for c in df.columns}  # Dict para detectar "Ticker"/"Bid"/"Ask" ignorando mayúsculas.
    if "ticker" in cols and "symbol" not in df.columns:  # Si aparece "Ticker" pero no "symbol".
        rename_map[cols["ticker"]] = "symbol"
    if "bid" in cols and "px_bid" not in df.columns:  # Si aparece "Bid" pero no "px_bid".
        rename_map[cols["bid"]] = "px_bid"
    if "ask" in cols and "px_ask" not in df.columns:  # Si aparece "Ask" pero no "px_ask".
        rename_map[cols["ask"]] = "px_ask"
    out = df.rename(columns=rename_map)  # Aplica renombres.
    for c in ["symbol", "px_bid", "px_ask"]:  # Asegura que existan las columnas clave.
        if c not in out.columns:
            out[c] = np.nan
    # Reordena con las 3 primeras columnas estándar + el resto.
    return out[["symbol", "px_bid", "px_ask"] + [c for c in out.columns if c not in ["symbol", "px_bid", "px_ask"]]]

@st.cache_data(ttl=90, show_spinner=False)  # Cachea el DataFrame combinado 90s.
def build_df_all():
    # Descarga las 3 fuentes en paralelo (I/O-bound), las normaliza y concatena.
    urls = [URL_BONDS, URL_NOTES, URL_CORPS]  # Lista de endpoints.
    dfs = []  # Acumula DataFrames normalizados.
    with ThreadPoolExecutor(max_workers=3) as ex:  # Pool de hilos para I/O.
        futs = {ex.submit(fetch_json, u): u for u in urls}  # Lanza 3 requests concurrentes.
        for fut in as_completed(futs):  # A medida que terminan…
            try:
                payload = fut.result()  # Obtiene el JSON.
                df = to_df(payload)  # Pasa a DataFrame plano.
                if not df.empty:  # Si hay datos…
                    dfs.append(harmonize_prices(df))  # Normaliza nombres y agrega a la lista.
            except Exception:
                pass  # Silencia este endpoint si falló; seguimos con los otros.
    if not dfs:  # Si no vino nada, devolver DF vacío estándar.
        return pd.DataFrame(columns=["symbol", "px_bid", "px_ask"])
    df_all = pd.concat(dfs, ignore_index=True, sort=False).drop_duplicates(subset=["symbol"])  # Concat y de-dup por símbolo.
    for c in ["px_bid", "px_ask"]:  # Enforce numérico en precios.
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    return df_all  # DataFrame listo para lookup de precios.

def get_price_for_symbol(df_all, symbol, prefer="px_ask"):
    # Busca precio por símbolo y prioriza columna 'prefer' (px_ask o px_bid); usa la alternativa si falta.
    row = df_all.loc[df_all["symbol"] == symbol]  # Filtra por símbolo exacto.
    if row.empty:  # Si no hay coincidencias, error claro.
        raise KeyError(f"No encontré {symbol} en df_all['symbol']")
    if prefer in row.columns and pd.notna(row.iloc[0][prefer]):  # Si hay precio en prefer…
        return float(row.iloc[0][prefer])  # Devuelve ese valor.
    alt = "px_bid" if prefer == "px_ask" else "px_ask"  # Alternativa.
    if alt in row.columns and pd.notna(row.iloc[0][alt]):  # Si la alternativa está presente…
        return float(row.iloc[0][alt])  # Devuelve alt.
    raise KeyError(f"{symbol}: no hay {prefer} ni {alt} con precio válido")  # Error si no hay ninguno.

# =====================================
# Parseo de celdas/fechas/números del Excel
# =====================================

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # Regex de fecha ISO simple YYYY-MM-DD.

def parse_date_cell(s):
    # Convierte una celda (varios formatos posibles) a datetime.
    if pd.isna(s):  # NaN -> None.
        return None
    if isinstance(s, (datetime, pd.Timestamp)):  # Si ya es fecha -> pydatetime.
        return pd.Timestamp(s).to_pydatetime()
    s = str(s).strip().replace("\u00A0", " ")  # Limpia espacios no separables.
    token = s.split("T")[0].split()[0]  # Recorta si viene "YYYY-MM-DDTHH:MM" o "fecha hora".
    if ISO_DATE_RE.match(token):  # Si matchea ISO…
        return datetime.strptime(token, "%Y-%m-%d")  # Parseo ISO.
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):  # Prueba formatos comunes d/m/Y, d-m-Y, Y/m/d.
        try:
            return datetime.strptime(token, fmt)  # Devuelve en el primer match válido.
        except ValueError:
            pass
    return pd.to_datetime(token, dayfirst=True, errors="raise").to_pydatetime()  # Fallback robusto (dayfirst=True).

def parse_date_list(cell):
    # Toma "2/3/2025;2/9/2025;..." y devuelve ["YYYY-MM-DD", ...].
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []  # Celda vacía -> lista vacía.
    parts = str(cell).replace(",", "/").split(";")  # Soporta coma como separador de fecha -> reemplaza por "/".
    out = []  # Lista de strings ISO.
    for p in parts:
        d = parse_date_cell(p)  # Parseo a datetime.
        out.append(d.strftime("%Y-%m-%d"))  # ISO string.
    return out  # Lista de fechas ISO.

def parse_float_cell(x):
    # Toma strings con "." y "," mezclados (1.234,56 o 1,234.56) y devuelve float o NaN.
    if pd.isna(x):
        return np.nan  # NaN si la celda está vacía.
    s = str(x).strip().replace("%", "")  # Saca el % si viniera.
    if "," in s and "." in s:  # Caso "1.234,56" -> quita puntos de miles y usa coma como decimal.
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")  # Caso simple "123,45".
    try:
        return float(s)  # Parseo final.
    except Exception:
        return np.nan  # Si no se puede, devuelve NaN.

def normalize_rate_to_percent(r):
    # Asegura que el cupón quede en porcentaje (si viene 0.25 -> 25).
    if pd.isna(r):
        return np.nan
    r = float(r)
    return r * 100.0 if r < 1 else r  # Si es <1 lo interpreta como proporción y lo pasa a %.

def parse_amorts(cell):
    # Convierte "10;10;10" -> [10.0, 10.0, 10.0]; celda vacía -> [].
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []  # Lista vacía si no hay amortizaciones.
    return [parse_float_cell(p) for p in str(cell).split(";")]  # Parseo elemento a elemento.

# =====================================
# Clase de ON (con memoización + XIRR rápido)
# =====================================

class ons_pro:
    def __init__(self, name, empresa, curr, law, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price):
        self.name = name  # Ticker del instrumento.
        self.empresa = empresa  # Emisor/empresa.
        self.curr = curr  # Moneda de pago.
        self.law = law  # Ley (local/NY, etc.).
        self.start_date = start_date  # Fecha de emisión (datetime).
        self.end_date = end_date  # Fecha de vencimiento final (datetime).
        self.payment_frequency = int(payment_frequency)  # Frecuencia en meses (e.g., 3 -> trimestral).
        if self.payment_frequency <= 0:  # Validación de frecuencia positiva.
            raise ValueError(f"{name}: payment_frequency debe ser > 0")
        self.amortization_dates = amortization_dates  # Lista de strings "YYYY-MM-DD" de amortizaciones de capital.
        self.amortizations = amortizations  # Lista de montos de amortización (por 100VN).
        self.rate = float(rate) / 100.0  # Cupón nominal anual en decimal (25 -> 0.25).
        self.price = float(price)  # Precio clean por 100VN.
        self._memo = {}  # Diccionario para cachear (schedule, arrays, etc.).

    def _freq(self):
        # Cantidad de cupones por año = 12 / meses_entre_cupones (redondeado y con piso 1).
        return max(1, int(round(12 / self.payment_frequency)))

    def _as_dt(self, d):
        # Asegura datetime (si viene string, parsea).
        return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")

    def _schedule_dt(self):
        # Genera y cachea el cronograma de fechas (datetime) desde settlement hasta maturity.
        if "schedule_dt" in self._memo:  # Si ya está en memo, devolverlo.
            return self._memo["schedule_dt"]
        settlement = datetime.today() + timedelta(days=1)  # Settlement T+1 como base de descuento.
        back = []  # Fechas hacia atrás desde el final.
        cur = self._as_dt(self.end_date)  # Arranca en maturity.
        start = self._as_dt(self.start_date)  # Fecha de emisión (límite inferior).
        back.append(cur)  # Incluye maturity.
        while True:  # Retrocede por períodos iguales a la frecuencia en meses.
            prev = cur - relativedelta(months=self.payment_frequency)  # Resta meses exactos.
            if prev <= start:  # Si ya pasamos la emisión, cortamos.
                break
            back.append(prev)  # Agrega fecha anterior.
            cur = prev  # Avanza el cursor hacia atrás.
        sched = [settlement] + sorted([d for d in back if d > settlement])  # Filtra futuras y antepone settlement.
        self._memo["schedule_dt"] = sched  # Cachea la lista.
        return sched  # Devuelve lista de datetime.

    def generate_payment_dates(self):
        # Devuelve las fechas del cronograma como strings ISO (para mapear contra amortizaciones del Excel).
        return [d.strftime("%Y-%m-%d") for d in self._schedule_dt()]

    def outstanding_on(self, ref_date=None):
        # Capital remanente (por 100VN) a una fecha de referencia (default: settlement).
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)  # Default settlement T+1.
        ref_date = self._as_dt(ref_date)  # Asegura datetime.
        # Suma amortizaciones con fecha <= ref_date.
        paid = sum(a for d, a in zip(self.amortization_dates, self.amortizations) if self._as_dt(d) <= ref_date)
        return max(0.0, 100.0 - paid)  # No puede ser negativo.

    def accrued_interest(self, ref_date=None):
        # Interés devengado lineal entre período de cupón (base ACT/ACT aproximada por días reales/365).
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)  # Fecha de referencia.
        ref_date = self._as_dt(ref_date)  # Asegura datetime.
        dates = self._schedule_dt()  # Fechas datetime del cronograma.
        coup_dt = dates[1:]  # Fechas de cupón (excluye settlement).
        if not coup_dt:
            return 0.0  # Si no hay cupones futuros, no devenga.
        # Primer cupón posterior a ref_date.
        next_coupon = next((d for d in coup_dt if d > ref_date), None)
        if next_coupon is None:
            return 0.0  # Si no hay cupón futuro, no devenga.
        idx = coup_dt.index(next_coupon)  # Índice del cupón futuro.
        # Fecha de inicio de período = emisión o cupón anterior.
        period_start = (max(self._as_dt(self.start_date), next_coupon - relativedelta(months=self.payment_frequency))
                        if idx == 0 else coup_dt[idx - 1])
        base = self.outstanding_on(period_start)  # Capital base sobre el que se calcula el cupón.
        full_coupon = (self.rate / self._freq()) * base  # Monto del cupón completo del período.
        total_days = max(1, (next_coupon - period_start).days)  # Días totales del período.
        accrued_days = max(0, min((ref_date - period_start).days, total_days))  # Días transcurridos.
        return full_coupon * (accrued_days / total_days)  # Devengado proporcional.

    def parity(self, ref_date=None):
        # Paridad = precio / (capital remanente + devengado) * 100.
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)  # Default fecha.
        vt = self.outstanding_on(ref_date) + self.accrued_interest(ref_date)  # Valor teórico base.
        return float('nan') if vt == 0 else round(self.price / vt * 100.0, 2)  # Evita división por cero.

    def amortization_payments(self):
        # Devuelve lista de amortizaciones alineada con generate_payment_dates (posición 0 es 0).
        caps = []  # Lista a devolver.
        dates_iso = self.generate_payment_dates()  # Fechas como strings ISO.
        am = dict(zip(self.amortization_dates, self.amortizations))  # Dict fecha->monto.
        for d in dates_iso:
            caps.append(am.get(d, 0.0))  # Si no hay amortización en esa fecha, 0.0.
        return caps  # Lista con mismo largo que el cronograma.

    def coupon_payments(self):
        # Devuelve lista de cupones alineada a generate_payment_dates (posición 0 es 0).
        dates = self._schedule_dt()  # Fechas datetime del cronograma.
        coupons = [0.0]  # Posición 0 (settlement) no paga cupón.
        coupon_dates = dates[1:]  # Solo fechas futuras de cupón.
        f = self._freq()  # Cupones por año.
        for i, cdate in enumerate(coupon_dates):
            # Inicio del período: emisión o cupón anterior.
            period_start = (max(self._as_dt(self.start_date), cdate - relativedelta(months=self.payment_frequency))
                            if i == 0 else coupon_dates[i - 1])
            base = self.outstanding_on(period_start)  # Capital remanente al inicio de período.
            coupons.append((self.rate / f) * base)  # Cupón = tasa periódica * base.
        return coupons  # Lista alineada.

    def cash_flow(self):
        # Construye el vector de flujos: en t=0 -price (clean), luego cupón+amortización por fecha.
        cfs = []  # Lista de flujos.
        caps = self.amortization_payments()  # Amortizaciones alineadas.
        cpns = self.coupon_payments()  # Cupones alineados.
        for i in range(len(cpns)):
            cfs.append(-self.price if i == 0 else caps[i] + cpns[i])  # t=0: -precio; resto: cap+cupón.
        return cfs  # Lista de flujos.

    def _times_and_flows(self):
        # Vectoriza tiempos (en años) y flujos; cachea para reuso en XNPV, XIRR, duración y convexidad.
        if "tf" in self._memo:  # Si ya está cacheado, devolverlo.
            return self._memo["tf"]
        d0 = datetime.today() + timedelta(days=1)  # Base (settlement) para t=0.
        dates = self._schedule_dt()  # Fechas datetime del cronograma.
        cfs = self.cash_flow()  # Flujos escala 100VN.
        t_years = np.array([(dt - d0).days / 365.0 for dt in dates], dtype=float)  # Tiempos en años ACT/365 aprox.
        c_arr = np.array(cfs, dtype=float)  # Flujos en array numpy.
        self._memo["tf"] = (t_years, c_arr)  # Cachea.
        return self._memo["tf"]  # Devuelve (t, c).

    # ---- XNPV/XIRR rápidos (vectorizados + Newton con fallback) ----

    def xnpv_vec(self, r):
        # NPV vectorizado: sum(c / (1+r)^t). r en decimal anual.
        t, c = self._times_and_flows()  # Obtiene arrays vectorizados.
        return np.sum(c / (1.0 + r) ** t)  # Descuento por potencias vectorizadas.

    def dxnpv_vec(self, r):
        # Derivada de NPV respecto a r para Newton-Raphson.
        t, c = self._times_and_flows()  # Arrays.
        return np.sum(-t * c / (1.0 + r) ** (t + 1))  # Derivada analítica.

    def xirr(self):
        # Resuelve r tal que NPV(r)=0; primero Newton (rápido), si no converge, bisección (robusto).
        guess = 0.25  # Arranque razonable (25%).
        r = guess  # Inicializa r.
        for _ in range(12):  # Hasta 12 iteraciones de Newton.
            f = self.xnpv_vec(r)  # NPV en r.
            df = self.dxnpv_vec(r)  # Derivada en r.
            if not np.isfinite(f) or not np.isfinite(df) or df == 0:  # Si algo está mal, corta Newton.
                break
            step = f / df  # Paso de Newton.
            r_new = r - step  # Actualiza r.
            if r_new <= -0.999:  # Evita regiones no definidas ((1+r)≈0).
                r_new = (r - 0.999) / 2  # Trae r a zona válida.
            if abs(r_new - r) < 1e-10:  # Convergencia numérica.
                return round(r_new * 100.0, 2)  # Devuelve en % con 2 decimales.
            r = r_new  # Continúa iterando.
        # Fallback: bisección en rango amplio si Newton no cerró.
        lo, hi = -0.9, 5.0  # Rango [-90%, 500%].
        flo, fhi = self.xnpv_vec(lo), self.xnpv_vec(hi)  # Evalúa extremos.
        if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:  # Si no hay cambio de signo, no se garantiza raíz.
            return float('nan')  # Devuelve NaN.
        for _ in range(40):  # 40 iteraciones suelen bastar.
            m = 0.5 * (lo + hi)  # Punto medio.
            fm = self.xnpv_vec(m)  # NPV en m.
            if abs(fm) < 1e-12:  # Casi cero -> solución.
                return round(m * 100.0, 2)  # Devuelve %.
            if flo * fm <= 0:  # Cambio de signo en [lo, m] -> mueve hi.
                hi, fhi = m, fm
            else:  # Si no, la raíz está en [m, hi] -> mueve lo.
                lo, flo = m, fm
        return round(0.5 * (lo + hi) * 100.0, 2)  # Devuelve centro final si se agotaron iteraciones.

    def tna_180(self):
        # TNA a 180 días a partir de XIRR efectivo (aprox. semestral *2).
        irr = self.xirr() / 100.0  # Pasa a decimal.
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 2)  # ((1+r)^(180/365)-1) anualizado simple.

    def duration(self):
        # Duración de Macaulay (años).
        irr = self.xirr() / 100.0  # YTM en decimal.
        t, c = self._times_and_flows()  # Tiempos y flujos vectorizados.
        mask = (t > 0) & (c != 0)  # Ignora t=0 y flujos nulos.
        if not np.any(mask):  # Si no hay flujos futuros, NaN.
            return float('nan')
        pv = np.sum(c[mask] / (1 + irr) ** t[mask])  # Precio presente de flujos (sin el flujo de compra).
        if pv == 0 or np.isnan(pv):  # Evita división por cero.
            return float('nan')
        mac = np.
