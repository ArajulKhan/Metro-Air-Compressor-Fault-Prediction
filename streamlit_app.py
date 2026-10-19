import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model.pkl")
columns = joblib.load("columns.pkl")

st.title("Metro (Air Compressor) Fault Prediction")
st.markdown("Provide the following details")

def num(label):
    return st.number_input(label, value=None, format="%g")

TP2 = num("TP2 Pressure")
TP3 = num("TP3 Pressure")
H1 = num("H1 Pressure")
DV_pressure = num("DV Pressure")
Reservoirs = num("Reservoirs Pressure")
Motor_current = num("Motor Current")
Oil_temperature = num("Oil Temperature")

COMP = st.selectbox("Compressor Signal", [0, 1])
DV_eletric = st.selectbox("DV Electric Signal", [0, 1])
Towers = st.selectbox("Tower Signal", [0, 1])
MPG = st.selectbox("MPG Signal", [0, 1])
LPS = st.selectbox("LPS Signal", [0, 1])
Pressure_switch = st.selectbox("Pressure Switch Signal", [0, 1], index=1)
Oil_level = st.selectbox("Oil Level Signal", [0, 1], index=1)
Caudal_impulses = st.selectbox("Caudal Impulses Signal", [0, 1], index=1)

#prediction button - Raw data & inputdata
if st.button("Predict"):
    Raw_data = {
        'TP2' : TP2,
        'TP3' : TP3,
        'H1': H1,
        'DV_pressure' : DV_pressure,
        'Reservoirs' : Reservoirs,
        'Motor_current' : Motor_current,
        'Oil_temperature' : Oil_temperature,
        'COMP' : COMP,
        'DV_eletric' : DV_eletric,
        'Towers' : Towers,
        'MPG' : MPG,
        'LPS' : LPS,
        'Pressure_switch' : Pressure_switch,
        'Oil_level' : Oil_level,
        'Caudal_impulses' : Caudal_impulses
    }

    input_data = pd.DataFrame([Raw_data])

    if input_data.isna().any(axis=None):
        st.warning("Please fill in all numeric fields.")
        st.stop()

    input_data = input_data[columns]
    prob = model.predict_proba(input_data)[0][1]

    if prob >= 0.5:
        st.error(f"Warning! High risk of fault (probability {prob:.1%})")
    else:
        st.success(f"No fault detected (fault probability {prob:.1%})")

    