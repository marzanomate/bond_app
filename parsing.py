from __future__ import annotations
import re
from datetime import datetime
import numpy as np
import pandas as pd


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
