import streamlit as st
import numpy as np
import pickle

# LOAD MODEL
model = pickle.load(open("model.pkl", "rb"))

# PAGE CONFIG
st.set_page_config(page_title="Rice Yield Prediction", layout="centered")

st.title("🌾 Rice Yield Prediction App")
st.write("Masukkan data pertanian untuk memprediksi hasil panen padi")

# INPUT FORM
st.subheader("Input Data")

land_area = st.number_input("Land Area (hectares)", min_value=0.0, step=0.1)

soil_type = st.selectbox(
    "Soil Type",
    ["Clay", "Sandy", "Loamy"]
)

rice_variety = st.selectbox(
    "Rice Variety",
    ["Variety A", "Variety B", "Variety C"]
)

fertilizer = st.number_input(
    "Fertilizer Amount (kg)",
    min_value=0.0,
    step=1.0
)

planting_method = st.selectbox(
    "Planting Method",
    ["Manual", "Machine"]
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    step=1.0
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    step=1.0
)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=0.0,
    step=0.1
)

# ENCODING
def encode_input(soil, variety, method):
    soil_map = {
        "Clay": 0,
        "Sandy": 1,
        "Loamy": 2
    }

    variety_map = {
        "Variety A": 0,
        "Variety B": 1,
        "Variety C": 2
    }

    method_map = {
        "Manual": 0,
        "Machine": 1
    }

    return (
        soil_map[soil],
        variety_map[variety],
        method_map[method]
    )

# PREDICTION
if st.button("Predict Yield"):

    soil_enc, variety_enc, method_enc = encode_input(
        soil_type,
        rice_variety,
        planting_method
    )

    input_data = np.array([[
        land_area,
        soil_enc,
        variety_enc,
        fertilizer,
        method_enc,
        rainfall,
        humidity,
        temperature
    ]])

    prediction = model.predict(input_data)[0]

    # OUTPUT
    st.subheader("Prediction Result")
    st.success(
        f"Predicted harvest yield: {round(prediction, 2)} ton/ha"
    )

    # SIMPLE EXPLANATION
    st.subheader("Explanation")

    explanation = ""

    if fertilizer > 100:
        explanation += "High fertilizer usage. "

    if rainfall >= 100 and rainfall <= 300:
        explanation += "Optimal rainfall condition. "

    if humidity >= 60 and humidity <= 80:
        explanation += "Good humidity level. "

    if temperature >= 25 and temperature <= 32:
        explanation += "Suitable temperature for rice growth. "

    if explanation == "":
        explanation = (
            "Conditions are moderate, resulting in average yield prediction."
        )

    st.info(explanation)
