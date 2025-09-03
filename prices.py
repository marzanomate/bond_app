

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


# ---------- Parsers / normalización ----------
def parse_date_cell(s):
    if pd.isna(s):
        return None
    if isinstance(s, datetime):
        return s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return pd.to_datetime(s, dayfirst=True, errors="raise").to_pydatetime()

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

def normalize_rate_to_percent(rate_raw):
    if pd.isna(rate_raw):
        return np.nan
    r = float(rate_raw)
    return r * 100.0 if r < 1 else r

def parse_amorts(cell):
    if cell is None or (isinstance(cell, float) and np.isnan(cell)) or (isinstance(cell, str) and cell.strip() == ""):
        return []
    parts = str(cell).split(";")
    return [parse_float_cell(p) for p in parts]

# ---------- Precio desde df_all ----------
def get_price_for_symbol(df_all, symbol, prefer="px_bid"):
    row = df_all.loc[df_all["symbol"] == symbol]
    if row.empty:
        return np.nan
    prefer_col = prefer if prefer in row.columns else None
    alt_col = "px_ask" if prefer == "px_bid" else "px_bid"
    alt_col = alt_col if alt_col in row.columns else None

    if prefer_col and not pd.isna(row.iloc[0][prefer_col]):
        return float(row.iloc[0][prefer_col])
    if alt_col and not pd.isna(row.iloc[0][alt_col]):
        return float(row.iloc[0][alt_col])
    for c in row.columns:
        if c != "symbol":
            try:
                return float(row.iloc[0][c])
            except Exception:
                continue
    return np.nan

# ---------- Construcción de objetos desde Excel ----------
def load_ons_from_excel(path_xlsx, df_all, price_col_prefer="px_bid", ons_class=None):
    """
    Lee Excel con columnas:
    name, empresa, curr, law, start_date, end_date, payment_frequency,
    amortization_dates, amortizations, rate, fr
    y devuelve lista de objetos ons_pro (u otra clase compatible).
    """
    if ons_class is None:
        # Se importará dinámicamente en app.py (o pásala explícitamente)
        from app import ons_pro as ons_class  # si tu clase está en app.py
    raw = pd.read_excel(path_xlsx, dtype=str)

    required_cols = [
        "name","empresa","curr","law","start_date","end_date",
        "payment_frequency","amortization_dates","amortizations","rate","fr"
    ]
    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Faltan columnas en el Excel: {missing}")

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
                raise ValueError(f"{name}: inconsistencia amortizaciones ({am_dates} vs {am_amts})")

        rate_raw = parse_float_cell(r["rate"])
        rate_pct = normalize_rate_to_percent(rate_raw)
        fr = int(parse_float_cell(r["fr"]))

        price = get_price_for_symbol(df_all, name, prefer=price_col_prefer)

        b = ons_class(
            name=name, empresa=emp, curr=curr, law=law,
            start_date=start, end_date=end,
            payment_frequency=pay_freq,
            amortization_dates=am_dates, amortizations=am_amts,
            rate=rate_pct, price=price, fr=fr
        )
        bonds.append(b)
    return bonds

# ---------- Métricas ----------
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
            rows.append([
                b.name, b.empresa, b.curr, b.law, b.rate * 100, b.price,
                np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
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
