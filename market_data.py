from __future__ import annotations
import pandas as pd
import requests
from functools import lru_cache


@lru_cache(maxsize=8)
def _fetch_json(url: str) -> dict:
r = requests.get(url, timeout=20)
r.raise_for_status()
return r.json()


def _to_df(payload):
if isinstance(payload, dict):
for key in ("data", "results", "items", "bonds", "notes"):
if key in payload and isinstance(payload[key], list):
payload = payload[key]
break
return pd.json_normalize(payload)


def load_market_data():
url_bonds = "https://data912.com/live/arg_bonds"
url_notes = "https://data912.com/live/arg_notes"
url_corps = "https://data912.com/live/arg_corp"
url_mep = "https://data912.com/live/mep"


df_bonds = _to_df(_fetch_json(url_bonds)); df_bonds["source"] = "bonds"
df_notes = _to_df(_fetch_json(url_notes)); df_notes["source"] = "notes"
df_corps = _to_df(_fetch_json(url_corps)); df_corps["source"] = "corps"
df_mep = _to_df(_fetch_json(url_mep)); df_mep["source"] = "mep"


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
