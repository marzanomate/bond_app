from __future__ import annotations
import io
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from core.calculator import bond_calculator_pro
from core.parsing import parse_date_cell, parse_date_list, parse_float_cell, normalize_rate_to_percent
from core.market_data import get_price_for_symbol


EXCEL_URL = "https://raw.githubusercontent.com/marzanomate/bond_app/main/listado_ons.xlsx"


def load_bcp_from_excel(df_all: pd.DataFrame, adj: float = 1.0, price_col_prefer: str = "px_bid") -> list:
content = requests.get(EXCEL_URL, timeout=25).content
raw = pd.read_excel(io.BytesIO(content), dtype=str)


required = [
"name","empresa","curr","law","start_date","end_date",
"payment_frequency","amortization_dates","amortizations",
"rate","outstanding","calificación"
]
missing = [c for c in required if c not in raw.columns]
if missing:
raise ValueError(f"Faltan columnas en el Excel: {missing}")


out = []
for _, r in raw.iterrows():
name = str(r["name"]).strip()
emisor = str(r["empresa"]).strip()
curr = str(r["curr"]).strip()
law = str(r["law"]).strip()
start = parse_date_cell(r["start_date"]) or datetime.today()
end = parse_date_cell(r["end_date"]) or datetime.today()


pf_raw = parse_float_cell(r["payment_frequency"])
if pd.isna(pf_raw) or pf_raw <= 0: continue
pf = int(round(pf_raw))


am_dates = parse_date_list(r["amortization_dates"])
am_amts = [parse_float_cell(x) for x in str(r["amortizations"]).split(";")] if str(r["amortizations"]).strip() != "" else []
if len(am_dates) != len(am_amts):
if len(am_dates) == 1 and len(am_amts) == 0:
am_amts = [100.0]
elif len(am_dates) == 0 and len(am_amts) == 1:
am_dates = [end.strftime("%Y-%m-%d")]
else:
continue


rate_pct = normalize_rate_to_percent(parse_float_cell(r["rate"]))
try:
price = get_price_for_symbol(df_all, name, prefer=price_col_prefer) * adj
except Exception:
price = np.nan
outstanding = parse_float_cell(r["outstanding"])
calif = str(r["calificación"]).strip()


b = bond_calculator_pro(
name=name, emisor=emisor, curr=curr, law=law,
start_date=start, end_date=end, payment_frequency=pf,
amortization_dates=am_dates, amortizations=am_amts,
rate=rate_pct, price=price,
step_up_dates=[], step_up=[],
outstanding=outstanding, calificacion=calif
)
out.append(b)
return out
