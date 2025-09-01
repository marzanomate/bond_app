pip install alphacast
pip install QuantLib-Python

import streamlit as st
import scipy.optimize as optimize
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import scipy.optimize
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import QuantLib as ql
from datetime import date, datetime, timedelta
import requests
from bcraapi import estadisticascambiarias
from alphacast import Alphacast

url_bonds = "https://data912.com/live/arg_bonds"
url_notes = "https://data912.com/live/arg_notes"
url_corps = "https://data912.com/live/arg_corp"
url_mep = "https://data912.com/live/mep"

def fetch_json(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def to_df(payload):
    # Accept list or dict with a top-level list (e.g., "data", "results", "items", ...)
    if isinstance(payload, dict):
        for key in ("data", "results", "items", "bonds", "notes"):
            if key in payload and isinstance(payload[key], list):
                payload = payload[key]
                break
    return pd.json_normalize(payload)

# Fetch
data_bonds = fetch_json(url_bonds)
data_notes = fetch_json(url_notes)
data_corps = fetch_json(url_corps)
data_mep = fetch_json(url_mep)

# Normalize to DataFrames
df_bonds = to_df(data_bonds)
df_notes = to_df(data_notes)
df_corps = to_df(data_corps)
df_mep = to_df(data_mep)

# Tag the origin (optional but useful)
df_bonds["source"] = "bonds"
df_notes["source"] = "notes"
df_corps["source"] = "corps"
df_mep["source"] = "mep"

# --- Option 1: stack/union (most common) ---
df_all = pd.concat([df_bonds, df_notes, df_corps], ignore_index=True, sort=False)
mep = df_mep.loc[df_mep["ticker"] == "AL30", "bid"].iloc[0]

# (optional) if you want unique rows per symbol:
# df_all = df_all.drop_duplicates(subset=["symbol"], keep="first")

print(df_all.head())
print(df_all.shape)
