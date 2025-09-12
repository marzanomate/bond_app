import streamlit as st
st.set_page_config(page_title="Bonos HD", page_icon="💵", layout="wide")


st.title("Bonos HD")
st.write("Usá el menú de la izquierda para navegar las secciones.")


st.page_link("pages/1_📈_Bonos_HD.py", label="📈 Bonos HD", icon="📈")
st.page_link("pages/2_💸_Lecaps.py", label="💸 Lecaps", icon="💸")
st.page_link("pages/3_🧰_Otros.py", label="🧰 Otros", icon="🧰")
