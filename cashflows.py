from __future__ import annotations
import pandas as pd
from core.calculator import bond_calculator_pro




def build_cashflow_table(selected_bonds: list, mode: str, inputs: dict) -> pd.DataFrame:
rows = []
for b in selected_bonds:
dates = b.generate_payment_dates()[1:]
coupons = b.coupon_payments()[1:]
capitals = b.amortization_payments()[1:]


if mode == "Nominal":
nominal = float(inputs.get(b.name, 0) or 0) / 100
else:
user_in = inputs.get(b.name, {})
monto = float(user_in.get("monto", 0) or 0)
precio_manual = user_in.get("precio", None)
precio = precio_manual if precio_manual else b.price
nominal = (monto / precio) if (precio and precio == precio) else 0.0


for d, cpn, cap in zip(dates, coupons, capitals):
rows.append({
"Fecha": d,
"Ticker": b.name,
"Cupón": round(cpn * nominal, 2),
"Capital": round(cap * nominal, 2),
"Total": round((cpn + cap) * nominal, 2)
})


df = pd.DataFrame(rows)
if df.empty:
return pd.DataFrame(columns=["Fecha", "Cupón", "Capital", "Total"])


df_total = df.groupby("Fecha", as_index=False)[["Cupón", "Capital", "Total"]].sum()
df_total[["Cupón","Capital","Total"]] = df_total[["Cupón","Capital","Total"]].round(2)
return df_total




def clone_with_price(b: bond_calculator_pro, new_price: float) -> bond_calculator_pro:
return bond_calculator_pro(
name=b.name, emisor=b.emisor, curr=b.curr, law=b.law,
start_date=b.start_date, end_date=b.end_date,
payment_frequency=b.payment_frequency,
amortization_dates=b.amortization_dates, amortizations=b.amortizations,
rate=b.rate, price=new_price,
step_up_dates=b.step_up_dates, step_up=b.step_up,
outstanding=b.outstanding, calificacion=b.calificacion
)




def compare_metrics_three(bond_map, sel_names: list, prices: list) -> pd.DataFrame:
from core.metrics import metrics_bcp
clones = []
for n, p in zip(sel_names, prices):
if not n: continue
base = bond_map[n]
clones.append(clone_with_price(base, float(p)))
return metrics_bcp(clones)
