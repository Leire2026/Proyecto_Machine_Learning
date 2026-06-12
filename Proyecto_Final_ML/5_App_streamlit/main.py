import streamlit as st
import pickle
import pandas as pd

# Cargar modelo y scaler
with open("random_forest.pkl", "rb") as a:
    modelo = pickle.load(a)

with open("scaler.pkl", "rb") as a:
    scaler = pickle.load(a)


st.title("Predictor de nivel de bienestar, ¿cómo de satisfecho estás con tu vida?")

st.write("Introduce los valores en los siguientes campos para estimar si alcanzas un nivel alto de bienestar:")


with st.form(key="formulario"):

    satisf_relac_pers = st.number_input("Satisfacción con las relaciones personales (0-10)", min_value=0, max_value=10, value=0)
    sentido_propia_vida = st.number_input("Sentido de la propia vida (0-10)", min_value=0, max_value=10, value=0)
    satisf_econom_domest = st.number_input("Satisfacción con la economía doméstica (0-10)", min_value=0, max_value=10, value=0)
    valor_estado_animo = st.number_input("Estado de ánimo (0-10)", min_value=0, max_value=10, value=0)
    satisf_vivienda = st.number_input("Satisfacción con la vivienda (0-10)", min_value=0, max_value=10, value=0)
    satisf_tiempo_disp = st.number_input("Satisfacción con el tiempo disponible (0-10)", min_value=0, max_value=10, value=0)
    satisf_viv_y_entorno = st.number_input("Satisfacción con vivienda y entorno (0-10)", min_value=0, max_value=10, value=0)
    salud_percibida = st.number_input("Salud percibida (1-4)", min_value=1, max_value=4, value=0)
    confianza_personas = st.number_input("Confianza en las personas (0-10)", min_value=0, max_value=10, value=0)
    confianza_poderes_pub = st.number_input("Confianza en los poderes públicos (0-10)", min_value=0, max_value=10, value=0)

    boton = st.form_submit_button("CALCULAR NIVEL DE BIENESTAR")


if boton:

    datos = pd.DataFrame({
        "Satisf_relac_pers": [satisf_relac_pers],
        "Sentido_propia_vida": [sentido_propia_vida],
        "Satisf_econom_domest": [satisf_econom_domest],
        "Valor_Estado_animo": [valor_estado_animo],
        "Satisf_vivienda": [satisf_vivienda],
        "Satisf_tiempo_disp": [satisf_tiempo_disp],
        "Satisf_viv_y_entorno": [satisf_viv_y_entorno],
        "Salud_percibida": [salud_percibida],
        "Confianza_personas": [confianza_personas],
        "Confianza_poderes_pub": [confianza_poderes_pub]
    })

    datos_scaled = scaler.transform(datos)

    pred = modelo.predict(datos_scaled)[0]
    prob = modelo.predict_proba(datos_scaled)[0, 1]

    if pred == 1:
        st.success(f"Resultado: Presentas un nivel de bienestar alto. Probabilidad estimada: {prob*100:.1f}%")
    else:
        st.warning(f"Resultado: No presentas un nivel de bienestar alto. Probabilidad estimada de bienestar alto: {prob*100:.1f}%")