# app.py — Calculadora de ONs (versión optimizada y comentada línea por línea)

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
st.set_page_config(page_title="Calculadora ONs", layout="wide")  # Título y layout ancho

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
    r = requests.get(url, timeout=25)  # GET con timeout prudente
    r.raise_for_status()  # Lanza si hay error HTTP
    return r.content  # Devuelve bytes del archivo

@st.cache_data(ttl=90, show_spinner=False)
def fetch_json(url: str):
    """Descarga JSON (cacheado 90s)."""
    r = requests.get(url, timeout=25)  # GET de JSON
    r.raise_for_status()  # Error si HTTP != 200
    return r.json()  # Retorna payload JSON

# =====================================
# Normalización de respuestas de precios
# =====================================

def to_df(payload):
    """Convierte payload (dict/list) a DataFrame plano."""
    if isinstance(payload, dict):  # Si es dict, busca listas comunes
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]  # Usa la lista contenida
                break
    return pd.json_normalize(payload)  # Normaliza a columnas


def harmonize_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica columnas a ['symbol','px_bid','px_ask', ...]."""
    rename_map = {}  # Mapa de renombres segun columnas existentes
    cols = {c.lower(): c for c in df.columns}  # Índice case-insensitive
    if "ticker" in cols and "symbol" not in df.columns:
        rename_map[cols["ticker"]] = "symbol"  # ticker -> symbol
    if "bid" in cols and "px_bid" not in df.columns:
        rename_map[cols["bid"]] = "px_bid"  # bid -> px_bid
    if "ask" in cols and "px_ask" not in df.columns:
        rename_map[cols["ask"]] = "px_ask"  # ask -> px_ask
    out = df.rename(columns=rename_map)  # Aplica renombres
    for c in ["symbol", "px_bid", "px_ask"]:
        if c not in out.columns:
            out[c] = np.nan  # Asegura columnas presentes
    # Reordena dejando las 3 claves al frente
    return out[["symbol", "px_bid", "px_ask"] + [c for c in out.columns if c not in ["symbol","px_bid","px_ask"]]]


@st.cache_data(ttl=90, show_spinner=False)
def build_df_all() -> pd.DataFrame:
    """Descarga 3 endpoints en paralelo (IO-bound) y concatena precios."""
    urls = [URL_BONDS, URL_NOTES, URL_CORPS]  # Lista de endpoints
    frames = []  # Acumula dataframes homogenizados
    with ThreadPoolExecutor(max_workers=3) as ex:  # Thread pool para IO
        futs = {ex.submit(fetch_json, u): u for u in urls}  # Lanza peticiones concurrentes
        for fut in as_completed(futs):  # A medida que terminan
            try:
                df = to_df(fut.result())  # Parseo payload a DF
                if not df.empty:  # Si trae datos
                    frames.append(harmonize_prices(df))  # Homogeniza columnas
            except Exception:
                pass  # Ignora errores puntuales de un endpoint
    if not frames:  # Si ninguno respondió
        return pd.DataFrame(columns=["symbol","px_bid","px_ask"])  # DF vacío con columnas clave
    df_all = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=["symbol"])  # Concat y dedup
    for c in ["px_bid","px_ask"]:
        df_all[c] = pd.to_numeric(df_all[c], errors="coerce")  # A número
    return df_all  # Retorna precios unificados


def get_price_for_symbol(df_all: pd.DataFrame, symbol: str, prefer: str = "px_ask") -> float:
    """Busca precio preferido por símbolo; si no hay, prueba la alternativa."""
    row = df_all.loc[df_all["symbol"] == symbol]  # Filtra símbolo exacto
    if row.empty:
        raise KeyError(f"No encontré {symbol} en df_all['symbol']")  # Error si no está
    if prefer in row.columns and pd.notna(row.iloc[0][prefer]):  # Si hay preferido
        return float(row.iloc[0][prefer])  # Devuelve preferido
    alt = "px_bid" if prefer == "px_ask" else "px_ask"  # Alternativa
    if alt in row.columns and pd.notna(row.iloc[0][alt]):  # Si hay alt válida
        return float(row.iloc[0][alt])  # Devuelve alt
    raise KeyError(f"{symbol}: no hay {prefer} ni {alt} con precio válido")  # Si no hay precio

# =====================================
# Parseo de celdas del Excel
# =====================================

ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # Regex para YYYY-MM-DD


def parse_date_cell(s):
    """Convierte múltiples formatos de fecha en datetime."""
    if pd.isna(s):
        return None  # NaN -> None
    if isinstance(s, (datetime, pd.Timestamp)):
        return pd.Timestamp(s).to_pydatetime()  # Normaliza a datetime nativo
    s = str(s).strip().replace("\u00A0", " ")  # Limpia espacios no-break
    token = s.split("T")[0].split()[0]  # Quita tiempo si viene ISO con hora
    if ISO_DATE_RE.match(token):
        return datetime.strptime(token, "%Y-%m-%d")  # Formato ISO
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):  # Intenta formatos comunes
        try:
            return datetime.strptime(token, fmt)  # Parse exitoso
        except ValueError:
            pass  # Prueba siguiente formato
    return pd.to_datetime(token, dayfirst=True, errors="raise").to_pydatetime()  # Fallback robusto


def parse_date_list(cell):
    """Split por ';' y parsea fechas a strings YYYY-MM-DD."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []  # Vacíos -> lista vacía
    parts = str(cell).replace(",", "/").split(";")  # Reemplaza coma por '/' y separa por ';'
    out = []  # Acumula fechas parseadas
    for p in parts:
        d = parse_date_cell(p)  # Parsea cada item
        out.append(d.strftime("%Y-%m-%d"))  # Normaliza a texto ISO
    return out  # Retorna lista de fechas ISO


def parse_float_cell(x):
    """Convierte strings con comas/puntos/porcentajes a float."""
    if pd.isna(x):
        return np.nan  # NaN queda NaN
    s = str(x).strip().replace("%", "")  # Quita % si aparece
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")  # Formato 1.234,56 -> 1234.56
    else:
        s = s.replace(",", ".")  # Reemplaza coma decimal
    try:
        return float(s)  # A float
    except Exception:
        return np.nan  # Falla -> NaN


def normalize_rate_to_percent(r):
    """Si <1 asume decimal (0.12 -> 12%); si >=1 asume ya en % (12 -> 12%)."""
    if pd.isna(r):
        return np.nan  # NaN permanece
    r = float(r)  # A float
    return r * 100.0 if r < 1 else r  # Escala si viene en decimal


def parse_amorts(cell):
    """Parsea lista de amortizaciones separadas por ';' a floats."""
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []  # Vacío
    return [parse_float_cell(p) for p in str(cell).split(";")]  # Lista de floats

# =====================================
# Clase de ON (con memoización y XIRR vectorizado)
# =====================================

class ons_pro:
    """Modelo de ON con generación de flujos y métricas."""
    def __init__(self, name, empresa, curr, law, start_date, end_date, payment_frequency,
                 amortization_dates, amortizations, rate, price):
        self.name = name  # Ticker
        self.empresa = empresa  # Emisor
        self.curr = curr  # Moneda de pago
        self.law = law  # Ley aplicable
        self.start_date = start_date  # Fecha emisión (datetime)
        self.end_date = end_date  # Fecha vencimiento (datetime)
        self.payment_frequency = int(payment_frequency)  # Frecuencia en meses (p.ej. 6 -> semestral)
        if self.payment_frequency <= 0:
            raise ValueError(f"{name}: payment_frequency debe ser > 0")  # Validación
        self.amortization_dates = amortization_dates  # Fechas de amortización (YYYY-MM-DD)
        self.amortizations = amortizations  # Monto a amortizar por 100 nominal
        self.rate = float(rate) / 100.0  # Cupón nominal anual en decimal
        self.price = float(price)  # Precio clean por 100 nominal
        self._memo = {}  # Diccionario de memoización interna

    def _freq(self):
        """Número de cupones por año (12 / meses)."""
        return max(1, int(round(12 / self.payment_frequency)))  # Cupones/año

    def _as_dt(self, d):
        """Asegura tipo datetime de entradas (str -> datetime)."""
        return d if isinstance(d, datetime) else datetime.strptime(d, "%Y-%m-%d")  # Normaliza

    def outstanding_on(self, ref_date=None):
        """Saldo de capital remanente a una fecha (100 - amortizaciones pagadas)."""
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)  # Liquidación T+1 aprox
        ref_date = self._as_dt(ref_date)  # Normaliza fecha
        paid = sum(a for d, a in zip(self.amortization_dates, self.amortizations)
                   if self._as_dt(d) <= ref_date)  # Suma amortizaciones ya pagadas
        return max(0.0, 100.0 - paid)  # Saldo mínimo 0

    def _schedule(self):
        """Genera cronograma desde start->end (solo fechas futuras), memoizado."""
        if "schedule" in self._memo:  # Usa cache si existe
            return self._memo["schedule"]  # Retorna lista de datetimes
        settlement = datetime.today() + timedelta(days=1)  # Fecha de liquidación
        back = []  # Lista de fechas hacia atrás desde el vencimiento
        cur = self._as_dt(self.end_date)  # Fecha vencimiento
        start = self._as_dt(self.start_date)  # Fecha inicio
        back.append(cur)  # Incluye vencimiento
        while True:
            prev = cur - relativedelta(months=self.payment_frequency)  # Resta un período
            if prev <= start:
                break  # Detiene si cruza el inicio
            back.append(prev)  # Agrega fecha intermedia
            cur = prev  # Avanza
        schedule = [settlement] + sorted([d for d in back if d > settlement])  # Incluye settlement y futuras
        self._memo["schedule"] = schedule  # Guarda en memo
        return schedule  # Retorna lista

    def generate_payment_dates(self):
        """Fechas en texto ISO para UI/tablas."""
        return [d.strftime("%Y-%m-%d") for d in self._schedule()]  # Formatea a str

    def amortization_payments(self):
        """Vector de amortizaciones alineado a schedule (0 en fechas sin pago)."""
        cap = []  # Lista de capital a pagar en cada fecha
        dates = self.generate_payment_dates()  # Fechas como strings
        am = dict(zip(self.amortization_dates, self.amortizations))  # Mapa fecha->monto
        for d in dates:
            cap.append(am.get(d, 0.0))  # 0 si no existe pago ese día
        return cap  # Retorna lista

    def coupon_payments(self):
        """Cupones por período sobre saldo al inicio del período."""
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]  # Fechas datetime
        coupons = [0.0]  # Primer fila es la inversión (sin cupón)
        coupon_dates = dates[1:]  # Quita la fecha de hoy/settlement
        f = self._freq()  # Cupones/año
        for i, cdate in enumerate(coupon_dates):
            period_start = (max(self._as_dt(self.start_date),
                                cdate - relativedelta(months=self.payment_frequency))
                            if i == 0 else coupon_dates[i-1])  # Inicio del período
            base = self.outstanding_on(period_start)  # Saldo sobre el que aplica cupón
            coupons.append((self.rate / f) * base)  # Cupón proporcional
        return coupons  # Devuelve lista

    def residual_value(self):
        """Saldo remanente para cada fecha del schedule (para mostrar)."""
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]  # Fechas datetime
        return [self.outstanding_on(d) for d in dates]  # Lista de saldos

    def cash_flow(self):
        """Flujo de caja nominal (negativo en t0, positivos en pagos)."""
        cfs = []  # Lista de flujos
        dates = self.generate_payment_dates()  # Fechas ISO
        caps = self.amortization_payments()  # Amortizaciones
        cpns = self.coupon_payments()  # Cupones
        for i, _ in enumerate(dates):
            cfs.append(-self.price if i == 0 else caps[i] + cpns[i])  # -precio en t0, luego pagos
        return cfs  # Retorna flujos

    def _times_and_flows(self):
        """Vectoriza tiempos (años) y flujos para XNPV/XIRR; memoiza."""
        if "tf" in self._memo:  # Usa memo si existe
            return self._memo["tf"]  # Retorna tupla (t, c)
        d0 = datetime.today() + timedelta(days=1)  # Base de descuento
        dates = self._schedule()  # Fechas datetime
        caps = self.amortization_payments()  # Capitales
        cpns = self.coupon_payments()  # Cupones
        cfs = [-self.price] + [c + a for c, a in zip(cpns[1:], caps[1:])]  # Flujos alineados
        t_years = np.array([(dt - d0).days / 365.0 for dt in dates], dtype=float)  # Tiempos en años
        cfs_arr = np.array(cfs, dtype=float)  # Flujos como array
        self._memo["tf"] = (t_years, cfs_arr)  # Guarda en memo
        return self._memo["tf"]  # Retorna tupla

    def xnpv_vec(self, r):
        """Valor presente con rendimiento r (vectorizado)."""
        t, c = self._times_and_flows()  # Obtiene tiempos y flujos
        return float(np.sum(c / (1.0 + r) ** t))  # Suma de flujos descontados

    def dxnpv_vec(self, r):
        """Derivada de XNPV respecto de r (para Newton-Raphson)."""
        t, c = self._times_and_flows()  # Tiempos y flujos
        return float(np.sum(-t * c / (1.0 + r) ** (t + 1)))  # Derivada analítica

    def xirr(self):
        """TIR anualizada usando Newton-Raphson con bisección de respaldo."""
        guess = 0.25  # Suposición inicial (25%)
        r = guess  # Valor de trabajo
        for _ in range(12):  # Itera unas pocas veces
            f = self.xnpv_vec(r)  # Función a cero
            df = self.dxnpv_vec(r)  # Derivada
            if not np.isfinite(f) or not np.isfinite(df) or df == 0:  # Chequeos
                break  # Falla -> bisección
            r_new = r - f / df  # Paso de Newton
            if r_new <= -0.999:  # Evita división por cero (1+r)
                r_new = (r - 0.999) / 2  # Recorta
            if abs(r_new - r) < 1e-10:  # Convergencia numérica
                return round(r_new * 100.0, 2)  # Retorna en %
            r = r_new  # Actualiza
        # Respaldo: bisección rápida si Newton no convergió
        lo, hi = -0.9, 5.0  # Rango amplio
        flo, fhi = self.xnpv_vec(lo), self.xnpv_vec(hi)  # Evalúa extremos
        if np.isnan(flo) or np.isnan(fhi) or flo * fhi > 0:  # Sin raíz garantizada
            return float('nan')  # Retorna NaN
        for _ in range(40):  # Búsqueda binaria
            m = 0.5 * (lo + hi)  # Punto medio
            fm = self.xnpv_vec(m)  # Eval
            if abs(fm) < 1e-12:  # Casi cero
                return round(m * 100.0, 2)  # Retorna %
            if flo * fm <= 0:  # Raíz en [lo,m]
                hi, fhi = m, fm  # Actualiza tope
            else:  # Raíz en [m,hi]
                lo, flo = m, fm  # Actualiza piso
        return round(0.5 * (lo + hi) * 100.0, 2)  # Devuelve centro si no exacto

    def tna_180(self):
        """TNA equivalente a 180 días a partir de la TIR."""
        irr = self.xirr() / 100.0  # TIR en decimal
        return round((((1 + irr) ** 0.5 - 1) * 2) * 100.0, 2)  # Convierte a TNA semestral

    def duration(self):
        """Duración de Macaulay en años (promedio ponderado de plazos)."""
        irr = self.xirr() / 100.0  # TIR decimal
        d0 = datetime.today() + timedelta(days=1)  # Base
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]  # Fechas datetime
        cfs = self.cash_flow()  # Flujos nominales
        flows = [(cf, dt) for i, (cf, dt) in enumerate(zip(cfs, dates)) if i > 0 and cf != 0]  # Solo pagos futuros
        if not flows:
            return float('nan')  # Sin pagos -> NaN
        pv_price = sum(cf / (1 + irr) ** ((dt - d0).days / 365.0) for cf, dt in flows)  # Precio presente de pagos
        if pv_price == 0 or np.isnan(pv_price):
            return float('nan')  # Evita división por cero
        mac = sum(((dt - d0).days / 365.0) * (cf / (1 + irr) ** ((dt - d0).days / 365.0))
                  for cf, dt in flows) / pv_price  # Fórmula de Macaulay
        return round(mac, 2)  # Redondea a 2 decimales

    def modified_duration(self):
        """Duración modificada: sensibilidad del precio a cambios de tasa."""
        irr = self.xirr() / 100.0  # TIR
        dur = self.duration()  # Duración de Macaulay
        den = 1 + irr  # Denominador
        if den == 0 or np.isnan(den) or np.isnan(dur):
            return float('nan')  # Control de NaN
        return round(dur / den, 2)  # Duración modificada

    def convexity(self):
        """Convexidad: curvatura de precio vs tasa."""
        y = self.xirr() / 100.0  # TIR
        d0 = datetime.today() + timedelta(days=1)  # Base
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]  # Fechas
        cfs = self.cash_flow()  # Flujos
        flows = [(cf, (dt - d0).days / 365.0) for i, (cf, dt) in enumerate(zip(cfs, dates)) if i > 0 and cf != 0]  # Pagos
        if not flows:
            return float('nan')  # Sin pagos
        pv = sum(cf / (1 + y) ** t for cf, t in flows)  # Precio presente
        if pv <= 0 or np.isnan(pv):
            return float('nan')  # Control
        cx = sum(cf * t * (t + 1) / (1 + y) ** (t + 2) for cf, t in flows) / pv  # Fórmula de convexidad
        return round(cx, 2)  # Redondeo

    def current_yield(self):
        """Rendimiento corriente: cupones del año próximo sobre precio actual."""
        cpns = self.coupon_payments()  # Lista de cupones
        dates = [self._as_dt(s) for s in self.generate_payment_dates()]  # Fechas
        future_idx = [i for i, d in enumerate(dates)
                      if d > (datetime.today() + timedelta(days=1)) and cpns[i] > 0]  # Próximos pagos
        if not future_idx:
            return float('nan')  # No hay
        i0 = future_idx[0]  # Primer índice futuro
        n = min(self._freq(), len(cpns) - i0)  # Número de cupones en 1 año
        annual_coupons = sum(cpns[i0:i0 + n])  # Suma anual
        return round(annual_coupons / self.price * 100.0, 2)  # Porcentaje

    def parity(self, ref_date=None):
        """Paridad = Precio / (Valor Técnico) * 100."""
        if ref_date is None:
            ref_date = datetime.today() + timedelta(days=1)  # T+1
        vt = self.outstanding_on(ref_date) + self.coupon_payments()[0]  # VT ~ saldo (accrued ya está en cupón 0)
        return float('nan') if vt == 0 else round(self.price / vt * 100.0, 2)  # Paridad

# =====================================
# Carga de ONs desde Excel + métricas
# =====================================

@st.cache_data(show_spinner=False)
def load_ons_from_excel(path_or_bytes, df_all: pd.DataFrame, price_col_prefer: str = "px_ask"):
    """Lee el Excel, construye objetos ons_pro y les asigna precio."""
    required = [
        "name","empresa","curr","law","start_date","end_date",
        "payment_frequency","amortization_dates","amortizations","rate"
    ]  # Columnas obligatorias
    raw = pd.read_excel(path_or_bytes, dtype=str)  # Lee Excel como texto
    missing = [c for c in required if c not in raw.columns]  # Revisa columnas faltantes
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")  # Error si faltan

    bonds = []  # Lista de objetos ons_pro
    errors = []  # Mensajes de error por fila
    for _, r in raw.iterrows():  # Recorre filas
        try:
            name  = str(r["name"]).strip()  # Ticker
            emp   = str(r["empresa"]).strip()  # Emisor
            curr  = str(r["curr"]).strip()  # Moneda
            law   = str(r["law"]).strip()  # Ley
            start = parse_date_cell(r["start_date"])  # Fecha inicio -> datetime
            end   = parse_date_cell(r["end_date"])  # Fecha fin -> datetime

            pay_freq_raw = parse_float_cell(r["payment_frequency"])  # Frecuencia cruda
            if pd.isna(pay_freq_raw) or pay_freq_raw <= 0:  # Validación
                raise ValueError(f"{name}: payment_frequency inválido -> {r['payment_frequency']}")  # Error
            pay_freq = int(round(pay_freq_raw))  # Redondea a entero de meses

            am_dates = parse_date_list(r["amortization_dates"])  # Lista de fechas ISO
            am_amts  = parse_amorts(r["amortizations"])  # Lista de montos
            if len(am_dates) != len(am_amts):  # Si desbalanceadas
                if len(am_dates) == 1 and len(am_amts) == 0:
                    am_amts = [100.0]  # Una fecha -> amortiza 100%
                elif len(am_dates) == 0 and len(am_amts) == 1:
                    am_dates = [end.strftime("%Y-%m-%d")]  # Un monto -> asume vencimiento
                else:
                    raise ValueError(f"{name}: inconsistencia amortizaciones {am_dates} vs {am_amts}")  # Error

            rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))  # Normaliza cupón a %
            price    = get_price_for_symbol(df_all, name, prefer=price_col_prefer)  # Busca precio preferido

            b = ons_pro(  # Construye objeto de bono
                name=name, empresa=emp, curr=curr, law=law,
                start_date=start, end_date=end, payment_frequency=pay_freq,
                amortization_dates=am_dates, amortizations=am_amts,
                rate=rate_pct, price=price
            )
            bonds.append(b)  # Agrega a lista
        except Exception as e:  # Captura errores por fila
            errors.append(f"{r.get('name','?')}: {e}")  # Guarda mensaje
    if errors:  # Si hubo errores
        st.warning("Algunos bonos no se pudieron cargar:\n- " + "\n- ".join(errors))  # Muestra warning
    return bonds  # Retorna lista de objetos


def bond_fundamentals_ons(bond_objs: list[ons_pro]) -> pd.DataFrame:
    """Calcula métricas principales para una lista de ONs."""
    rows = []  # Acumulará filas de métricas
    for b in bond_objs:  # Itera bonos
        try:
            rows.append([  # Agrega fila con métricas
                b.name, b.empresa, b.curr, b.law, b.rate * 100, b.price,
                b.xirr(), b.tna_180(), b.duration(), b.modified_duration(),
                b.convexity(), b.current_yield(), b.parity()
            ])
        except Exception as e:  # Si métricas fallan
            rows.append([b.name, b.empresa, b.curr, b.law, b.price, np.nan,
                         np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])  # Fila con NaNs
            st.warning(f"⚠️ Error en {b.name}: {e}")  # Muestra aviso
    cols = [  # Nombres de columnas de salida
        "Ticker","Empresa","Moneda de Pago","Ley","Cupón","Precio",
        "Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"
    ]
    df = pd.DataFrame(rows, columns=cols)  # Construye DataFrame
    # Redondeos/formatos homogéneos
    for c in ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # A número
    df["Cupón"] = df["Cupón"].round(4)  # Cupón con 4 decimales
    df["Precio"] = df["Precio"].round(2)  # Precio 2 decimales
    for c in ["Yield","TNA_180","Current Yield","Paridad (%)"]:
        df[c] = df[c].round(2)  # Porcentajes 2 decimales
    for c in ["Dur","MD","Conv"]:
        df[c] = df[c].round(2)  # Métricas 2 decimales
    return df.reset_index(drop=True)  # Reindexa


def bond_flows_frame(b: ons_pro) -> pd.DataFrame:
    """Tabla de flujos y saldos por fechas para un bono."""
    dates = b.generate_payment_dates()  # Fechas ISO
    res   = b.residual_value()  # Saldo remanente
    caps  = b.amortization_payments()  # Capitales
    cpns  = b.coupon_payments()  # Cupones
    cfs   = b.cash_flow()  # Flujos netos
    return pd.DataFrame({  # Construye DataFrame tabular
        "Fecha": dates,
        "Residual": res,
        "Amortización": caps,
        "Cupón": cpns,
        "Flujo": cfs
    })

# =====================================
# UI
# =====================================

st.title("📈 Obligaciones Negociables")  # Título principal

# Botón actualizar precios
col_header = st.columns([1, 1, 6])  # Tres columnas para header
with col_header[0]:  # Columna del botón
    if st.button("🔄 Actualizar precios", type="primary", help="Refresca precios de data912", key="refresh_prices"):  # Botón refresh
        st.cache_data.clear()  # Limpia cachés de datos
        st.rerun()  # Fuerza rerun inmediato

# Carga de precios y excel
with st.spinner("Cargando precios"):  # Muestra spinner mientras se carga
    df_all = build_df_all()  # Descarga/lee precios (cacheado)
    if df_all.empty:  # Si no hay datos
        st.error("No hay precios disponibles")  # Error de datos
        st.stop()  # Detiene app
    try:
        excel_bytes = io.BytesIO(fetch_excel_bytes(EXCEL_URL_DEFAULT))  # Descarga Excel cacheado
    except Exception as e:  # Si falla la descarga
        st.error(f"No pude descargar el Excel desde la URL: {e}")  # Muestra error
        st.stop()  # Detiene app
    bonds = load_ons_from_excel(excel_bytes, df_all, price_col_prefer="px_ask")  # Crea objetos bonos
    if not bonds:  # Si lista vacía
        st.error("El Excel no produjo bonos válidos.")  # Error de contenido
        st.stop()  # Detiene app
    df_metrics = bond_fundamentals_ons(bonds)  # Calcula métricas

# =========================
# Filtros (en form para evitar rerun continuo)
# =========================

with st.form("filters"):  # Agrupa filtros en un formulario
    fc = st.columns(3)  # Tres columnas de filtros
    with fc[0]:  # Columna 1
        emp_opts = sorted(df_metrics["Empresa"].dropna().unique().tolist())  # Opciones de empresa
        sel_emp = st.multiselect("Empresa", emp_opts, default=emp_opts, key="filter_emp")  # Selección múltiple
    with fc[1]:  # Columna 2
        mon_opts = sorted(df_metrics["Moneda de Pago"].dropna().unique().tolist())  # Opciones moneda
        sel_mon = st.multiselect("Moneda de Pago", mon_opts, default=mon_opts, key="filter_mon")  # Selección múltiple
    with fc[2]:  # Columna 3
        ley_opts = sorted(df_metrics["Ley"].dropna().unique().tolist())  # Opciones ley
        sel_ley = st.multiselect("Ley", ley_opts, default=ley_opts, key="filter_ley")  # Selección múltiple
    submitted = st.form_submit_button("Aplicar filtros")  # Botón aplicar

# Si no se envió el formulario, asume todos seleccionados (evita lista vacía en primer render)
if not submitted:  # Primer render sin submit
    sel_emp = emp_opts  # Todas empresas
    sel_mon = mon_opts  # Todas monedas
    sel_ley = ley_opts  # Todas leyes

# Aplica filtros a la tabla de métricas
mask = (  # Condición booleana
    df_metrics["Empresa"].isin(sel_emp) &
    df_metrics["Moneda de Pago"].isin(sel_mon) &
    df_metrics["Ley"].isin(sel_ley)
)

df_view = df_metrics.loc[mask].reset_index(drop=True)  # Vista filtrada

# Define columnas numéricas para formateo consistente	num_cols = ["Cupón","Precio","Yield","TNA_180","Dur","MD","Conv","Current Yield","Paridad (%)"]  # Lista numérica

# Copia segura para formateo de la grilla
dfv = df_metrics.copy()  # Copia
for c in num_cols:
    dfv[c] = pd.to_numeric(dfv[c], errors="coerce")  # Asegura dtype float

# Muestra grilla de métricas
st.dataframe(  # Tabla interactiva
    dfv,  # DataFrame a mostrar
    hide_index=True,  # Sin índice
    use_container_width=True,  # Ocupa ancho
    column_config={  # Formatos por columna
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
csv = df_view.to_csv(index=False).encode("utf-8")  # Serializa CSV
st.download_button("⬇️ Descargar CSV", data=csv, file_name="ons_metrics.csv", mime="text/csv")  # Botón descarga

st.markdown("---")  # Separador visual

# =========================
# Flujos escalados
# =========================

colA, colB = st.columns([2, 3])  # Dos columnas sección flujos
with colA:  # Columna izquierda (controles)
    st.subheader("Flujos")  # Título
    tickers = ["(ninguno)"] + df_view["Ticker"].dropna().unique().tolist()  # Lista de tickers
    pick = st.selectbox("Ticker", tickers, index=0, key="flow_ticker")  # Selector de bono

    mode = st.radio(  # Modo de cálculo
        "Modo de cálculo",
        ["Por nominales (VN)", "Por monto / precio manual"],
        horizontal=False,
        key="flow_mode",
    )

    if mode == "Por nominales (VN)":  # Si selecciona por VN
        vn = st.number_input("Nominales (VN)", min_value=0.0, value=100.0, step=100.0, key="vn_input")  # VN
        precio_manual = None  # No aplica
        monto = None  # No aplica
    else:  # Si selecciona por monto
        monto = st.number_input("Monto a invertir", min_value=0.0, value=10000.0, step=1000.0, key="monto_input")  # Monto
        precio_manual = st.number_input(  # Precio manual por 100
            "Precio manual (por 100 nominal, clean)",
            min_value=0.0001, value=100.0, step=0.5, key="precio_manual_flows"
        )
        vn = None  # No aplica VN directo

with colB:  # Columna derecha (tabla flujos)
    if pick and pick != "(ninguno)":  # Si eligió ticker válido
        bmap = {b.name: b for b in bonds}  # Mapa name->objeto
        if pick in bmap:  # Existe en mapa
            b = bmap[pick]  # Objeto bono
            st.write(f"**{pick}** — Empresa: {b.empresa} · Moneda: {b.curr} · Ley: {b.law} · Cupón: {round(b.rate*100,4)}% · Precio (px_ask): {b.price:.2f}")  # Header

            df_flows = bond_flows_frame(b)  # DataFrame de flujos

            # Calcula escala (excluyendo fila 0)
            if mode == "Por nominales (VN)":  # Caso VN
                scale = (vn or 0.0) / 100.0  # Escala por 100
            else:  # Caso monto/precio
                scale = 0.0 if not monto or not precio_manual else (monto / precio_manual)  # Número de bloques de 100

            df_cash = df_flows.copy()  # Copia para escalar
            if scale > 0:  # Si escala válida
                df_cash.loc[1:, ["Residual","Amortización","Cupón","Flujo"]] = \
                    df_cash.loc[1:, ["Residual","Amortización","Cupón","Flujo"]].astype(float) * scale  # Escala columnas

            st.dataframe(df_cash.iloc[1:].reset_index(drop=True), use_container_width=True, height=360)  # Muestra tabla sin t0

            if scale > 0:  # Si hubo escala
                total_cobros = float(df_cash["Flujo"].iloc[1:].sum())  # Suma de flujos futuros
                st.metric("Total a cobrar (sumatoria flujos futuros)", f"{total_cobros:,.2f}")  # Muestra métrica
        else:  # No hallado
            st.warning(f"No encontré el bono {pick} en la lista cargada.")  # Warning

st.markdown("---")  # Separador

# =========================
# Métricas con precio manual
# =========================

st.subheader("Calculadora de métricas")  # Título sección
colM1, colM2, colM3, colM4 = st.columns([2, 1.2, 1.2, 3])  # Cuatro columnas

# Fila de etiquetas para alinear visualmente
with colM1: st.markdown("**Ticker**")  # Etiqueta visible
with colM2: st.markdown("**Precio manual**")  # Etiqueta visible
with colM3: st.markdown("** **")  # Espaciador
with colM4: st.markdown("**Resultado**")  # Etiqueta visible

# Widgets con labels colapsadas (quedan en una misma fila)
with colM1:
    tick2 = st.selectbox(  # Selector de bono
        label="Ticker",
        options=["(ninguno)"] + df_metrics["Ticker"].dropna().unique().tolist(),
        index=0,
        key="manual_ticker",
        label_visibility="collapsed",
    )

with colM2:
    pman = st.number_input(  # Precio manual de evaluación
        label="Precio manual",
        min_value=0.0, value=100.0, step=0.5,
        key="manual_price",
        label_visibility="collapsed",
    )

with colM3:
    go_btn = st.button("Calcular métricas", key="calc_metrics_btn", type="primary", use_container_width=True)  # Botón acción

with colM4:
    if go_btn and tick2 and tick2 != "(ninguno)":  # Si clic y ticker válido
        bmap = {b.name: b for b in bonds}  # Mapa name->objeto
        if tick2 in bmap:  # Existe
            b0 = bmap[tick2]  # Bono base
            # Clona con precio manual (rate en % porque __init__ divide entre 100)
            b = ons_pro(
                name=b0.name, empresa=b0.empresa, curr=b0.curr, law=b0.law,
                start_date=b0.start_date, end_date=b0.end_date, payment_frequency=b0.payment_frequency,
                amortization_dates=b0.amortization_dates, amortizations=b0.amortizations,
                rate=b0.rate*100.0,  # __init__ espera %
                price=pman  # Precio manual
            )
            df_one = bond_fundamentals_ons([b])  # Calcula métricas
            st.dataframe(df_one, use_container_width=True, height=120, hide_index=True)  # Muestra
        else:
            st.warning(f"No encontré el bono {tick2} para el cálculo manual.")  # Warning
