from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime
from core.calculator import bond_calculator_pro


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
"Current Yield": b.current_yield(stl),
"Calificación": b.calificacion,
"Próxima Fecha de Pago": prox,
"Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d"),
})
except Exception:
rows.append({
"Ticker": getattr(b, "name", np.nan),
"Emisor": getattr(b, "emisor", np.nan),
"Ley": getattr(b, "law", np.nan),
"Moneda de Pago": getattr(b, "curr", np.nan),
"Precio": round(getattr(b, "price", np.nan), 1) if hasattr(b, "price") else np.nan,
"TIR": np.nan, "TNA SA": np.nan, "Modified Duration": np.nan,
"Duration": np.nan, "Convexidad": np.nan, "Paridad": np.nan,
"Current Yield": np.nan,
"Calificación": getattr(b, "calificacion", np.nan),
"Próxima Fecha de Pago": None,
"Fecha de Vencimiento": b.end_date.strftime("%Y-%m-%d") if hasattr(b, "end_date") else None,
})


df = pd.DataFrame(rows)
for c in ["TIR","TNA SA","Paridad","Current Yield"]:
df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
for c in ["Duration","Modified Duration","Convexidad","Precio"]:
df[c] = pd.to_numeric(df[c], errors="coerce").round(1)
return df.reset_index(drop=True)
