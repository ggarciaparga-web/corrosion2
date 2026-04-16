import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import erf

# Configuración de la página
st.set_page_config(page_title="Visor de Corrosión", layout="wide")

# ---------------------------------------------------------
# BARRA LATERAL: INPUTS GLOBALES
# ---------------------------------------------------------
st.sidebar.header("Configuración General")

with st.sidebar.expander("Tiempo y Corrosión", expanded=True):
    t_end = st.number_input("Tiempo final (años)", value=100)
    time_step = st.number_input("Paso de tiempo (años)", value=1)
    i_corr = st.slider("i_corr (µA/cm²)", 0.0, 10.0, 2.0)
    alpha = st.slider("Factor de ataque (alpha)", 1, 30, 10)

with st.sidebar.expander("Geometría y Armaduras", expanded=True):
    b_initial = st.number_input("Ancho b (mm)", value=150)
    d_initial = st.number_input("Canto útil d (mm)", value=300)
    cover = st.number_input("Recubrimiento (mm)", value=20)
    phi1_0 = st.number_input("Diámetro barras inf. (mm)", value=20.0)
    n_bottom = st.number_input("Número de barras inf.", value=2)
    phi_w0 = st.number_input("Diámetro estribos (mm)", value=0.0001, format="%.4f")
    r2 = st.number_input("r2 para Mu Cons (mm)", value=20)

with st.sidebar.expander("Materiales", expanded=True):
    fck = st.number_input("fck (MPa)", value=25)
    fy = st.number_input("fy (MPa)", value=500)

# Parámetros derivados
fyd = fy / 1.15
fci = 0.333 * fck ** (2 / 3)
fcd_nominal = fck / 1.5

# ---------------------------------------------------------
# FUNCIONES DE CÁLCULO
# ---------------------------------------------------------

def run_model_code():
    times = np.arange(0, t_end + time_step, time_step)
    results = []
    
    # Reducción fib 2023
    nfc = min(1.0, (30.0 / fck) ** (1.0 / 3.0))
    kc = 0.75 * nfc
    fcd_reduced = kc * fcd_nominal

    for t in times:
        px = 0.0116 * i_corr * t
        p1 = max(0.0, phi1_0 - alpha * px)
        as_corr = (math.pi * p1 ** 2 / 4.0) * n_bottom
        
        # Mu standard
        x_std = (as_corr * fyd) / (0.8 * b_initial * fcd_reduced)
        mu_std = (as_corr * fyd * (d_initial - 0.4 * x_std)) / 1e6 if p1 > 0 else 0
        
        # Mu Cons
        z_cons = max(0, d_initial - r2 - 0.4 * x_std)
        mu_cons = (as_corr * fyd * z_cons) / 1e6 if p1 > 0 else 0
        
        results.append({"Time": t, "Px": px, "phi": p1, "As": as_corr, "Mu (kNm)": mu_std, "Mu Cons (kNm)": mu_cons})
    return pd.DataFrame(results)

def run_contevect():
    times = np.arange(0, t_end + time_step, time_step)
    px0 = max(0.0, (83.8 + 7.4 * (cover / phi1_0) - 22.6 * fci) * 1e-3)
    
    # Simulación para detectar eventos
    rows_base = []
    for t in times:
        px = 0.0116 * i_corr * t
        p1 = max(0.0, phi1_0 - alpha * px)
        a1 = (np.pi * p1 ** 2 / 4.0) * n_bottom
        rows_base.append({"t": t, "px": px, "a1": a1, "rho1": a1/(b_initial*d_initial)})
    
    df_b = pd.DataFrame(rows_base)
    
    # Lógica de estados críticos (Spalling)
    b_act, d_act = b_initial, d_initial
    final_data = []
    
    for _, r in df_b.iterrows():
        # Simplificación de lógica CONTEVECT para el visor
        if r['px'] >= px0 and r['px'] < 0.2:
            d_act = d_initial # Todavía no hay spalling agresivo
        elif r['px'] >= 0.2:
            b_act = b_initial - 2 * cover
            d_act = d_initial - cover
            
        mu = (r['a1'] * fyd * (d_act - 0.4 * (r['a1']*fyd/(0.8*b_act*(fck/1.5))))) / 1e6
        final_data.append({"Tiempo": r['t'], "Px": r['px'], "b": b_act, "d": d_act, "Mu (kNm)": max(0, mu)})
        
    return pd.DataFrame(final_data)

# ---------------------------------------------------------
# INTERFAZ DE USUARIO (TABS)
# ---------------------------------------------------------
st.title("🛡️ Comparador de Modelos de Corrosión")

tab1, tab2, tab3 = st.tabs(["📊 Model Code", "🏗️ CONTEVECT", "🔄 Comparativa"])

df_m1 = run_model_code()
df_m2 = run_contevect()

with tab1:
    st.header("Análisis: Model Code (fib)")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(df_m1.style.format("{:.2f}"))
    fig1, ax1 = plt.subplots()
    ax1.plot(df_m1["Time"], df_m1["Mu (kNm)"], label="Mu Standard", linewidth=2)
    ax1.plot(df_m1["Time"], df_m1["Mu Cons (kNm)"], '--', label="Mu Cons")
    ax1.set_xlabel("Años")
    ax1.set_ylabel("Capacidad (kNm)")
    ax1.legend()
    c2.pyplot(fig1)

with tab2:
    st.header("Análisis: CONTEVECT (Spalling)")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(df_m2.style.format("{:.2f}"))
    fig2, ax2 = plt.subplots()
    ax2.plot(df_m2["Tiempo"], df_m2["Mu (kNm)"], color="orange", marker="o", markersize=2)
    ax2.set_xlabel("Años")
    ax2.set_ylabel("Capacidad (kNm)")
    c2.pyplot(fig2)

with tab3:
    st.header("Comparativa Directa")
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    ax3.plot(df_m1["Time"], df_m1["Mu (kNm)"], label="Model Code", color="blue")
    ax3.plot(df_m2["Tiempo"], df_m2["Mu (kNm)"], label="CONTEVECT", color="orange", linestyle="--")
    ax3.set_xlabel("Años")
    ax3.set_ylabel("Mu (kNm)")
    ax3.legend()
    st.pyplot(fig3)