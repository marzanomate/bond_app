import streamlit as st
st.subheader("Comparador de Métricas (3 bonos)")
choices = sorted(name_to_bond.keys())
col1, col2, col3 = st.columns(3)


with col1:
b1_name = st.selectbox("Bono 1", choices, index=0, key="cmp_b1")
p1 = st.number_input("Precio manual 1 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p1")
with col2:
b2_name = st.selectbox("Bono 2", choices, index=1, key="cmp_b2")
p2 = st.number_input("Precio manual 2 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p2")
with col3:
b3_name = st.selectbox("Bono 3", choices, index=2, key="cmp_b3")
p3 = st.number_input("Precio manual 3 (opcional)", min_value=0.0, step=0.1, value=0.0, key="cmp_p3")


if st.button("Calcular comparativa"):
df_cmp = compare_metrics_three({b.name: b for b in all_bonds}, [b1_name, b2_name, b3_name], [p1, p2, p3])
st.dataframe(
df_cmp.style.format({
"Precio": "{:.1f}", "TIR": "{:.1f}", "TNA SA": "{:.1f}",
"Duration": "{:.1f}", "Modified Duration": "{:.1f}",
"Convexidad": "{:.1f}", "Paridad": "{:.1f}", "Current Yield": "{:.1f}",
}),
use_container_width=True, hide_index=True
)


st.divider()


# =========================
# 4) Curvas comparadas por Emisor
# =========================
st.subheader("Curvas comparadas por Emisor (TIR vs Modified Duration)")
df_metrics = metrics_bcp(all_bonds).copy()


emisores_all = sorted([e for e in df_metrics["Emisor"].dropna().unique()])
colc1, colc2, colc3 = st.columns([1,1,2])
with colc1:
em1 = st.selectbox("Emisor A", emisores_all, index=0, key="curve_em1")
with colc2:
idx_default = 1 if len(emisores_all) > 1 else 0
em2 = st.selectbox("Emisor B", emisores_all, index=idx_default, key="curve_em2")
with colc3:
st.caption("Gráfico: eje X = Modified Duration | eje Y = TIR (e.a. %)")


emisores_sel = [em1, em2] if em1 != em2 else [em1]
df_curves = df_metrics[df_metrics["Emisor"].isin(emisores_sel)].copy()


for c in ["TIR", "Modified Duration", "Precio", "TNA SA", "Convexidad", "Paridad", "Current Yield"]:
if c in df_curves.columns:
df_curves[c] = pd.to_numeric(df_curves[c], errors="coerce")


if not df_curves.empty:
fig = px.scatter(
df_curves, x="Modified Duration", y="TIR", color="Emisor", symbol="Emisor",
hover_name="Ticker",
hover_data={
"Emisor": True, "Ticker": False, "Ley": True, "Moneda de Pago": True,
"Precio": ":.1f", "TIR": ":.1f", "TNA SA": ":.1f", "Modified Duration": ":.1f",
"Convexidad": ":.1f", "Paridad": ":.1f", "Current Yield": ":.1f",
}, size_max=12,
)
fig.update_traces(marker=dict(size=12, line=dict(width=1)))
fig.update_layout(
xaxis_title="Modified Duration (años)", yaxis_title="TIR (%)",
legend_title="Emisor", height=480, margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig, use_container_width=True)


st.markdown("**Bonos incluidos en las curvas:**")
cols_show = [
"Ticker","Emisor","Ley","Moneda de Pago","Precio",
"TIR","TNA SA","Modified Duration","Convexidad","Paridad","Current Yield",
"Próxima Fecha de Pago","Fecha de Vencimiento"
]
cols_show = [c for c in cols_show if c in df_curves.columns]
st.dataframe(
df_curves[cols_show].style.format({
"Precio": "{:.1f}", "TIR": "{:.1f}", "TNA SA": "{:.1f}",
"Modified Duration": "{:.1f}", "Convexidad": "{:.1f}",
"Paridad": "{:.1f}", "Current Yield": "{:.1f}",
}), use_container_width=True, hide_index=True
)
else:
st.info("No hay bonos para los emisores seleccionados.")
