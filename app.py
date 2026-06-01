import pickle
import streamlit as st

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
    
import pandas as pd
import joblib
import numpy as np

@st.cache_resource
def load_model():
    model = joblib.load('random_forest_model.pkl')
    scaler = joblib.load('scaler.pkl')
    columns = joblib.load('model_columns.pkl')
    return model, scaler, columns

model, scaler, model_columns = load_model()

st.set_page_config(page_title="Prediksi Panen Padi", layout="centered")
st.title("Prediksi Hasil Panen Padi")
st.markdown("**Random Forest Regressor** | 8 Fitur Utama")

st.sidebar.header("Input Data Lahan")

hectares = st.sidebar.number_input("Luas Lahan (Hectares)", min_value=0.1, value=2.0, step=0.1)

variety = st.sidebar.selectbox(
    "Varietas Padi", 
    options=['CO_43', 'ponmani', 'delux ponni']
)

soil_type = st.sidebar.selectbox(
    "Jenis Tanah", 
    options=['alluvial', 'clay']
)

planting_method = st.sidebar.selectbox(
    "Metode Tanam",
    options=['manual', 'machine']
)

temperature = st.sidebar.number_input(
    "Suhu (°C)",
    min_value=0.0,
    value=30.0,
    step=0.1
)

seedrate = st.sidebar.number_input("Seedrate (Kg)", min_value=10, value=50, step=5)

urea_40 = st.sidebar.number_input("Urea 40 Hari (Kg)", min_value=0.0, value=54.26, step=0.1)

potash_50 = st.sidebar.number_input("Potash 50 Hari (Kg)", min_value=0.0, value=20.76, step=0.1)

rain_30d = st.sidebar.number_input("Curah Hujan 30 Hari Pertama (mm)", min_value=0.0, value=18.5, step=0.1)

humidity_30d = st.sidebar.number_input("Kelembaban Rata-rata D1-D30 (%)", min_value=0.0, value=72.0, step=0.1)

if st.sidebar.button("Prediksi Hasil Panen"):
    input_data = pd.DataFrame({
    'Hectares': [hectares],
    'Variety': [variety],
    'Soil Types': [soil_type],
    'Seedrate(in Kg)': [seedrate],
    'Urea_40Days': [urea_40],
    'Potassh_50Days': [potash_50],
    '30DRain( in mm)': [rain_30d],
    'Relative Humidity_D1_D30': [humidity_30d],

    # Tambahan
    'Planting Method': [planting_method],
    'Temperature': [temperature]
})

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_encoded)

    prediction_kg = model.predict(input_scaled)[0]

    if hectares > 0:
        prediction_ton_per_ha = (prediction_kg / 1000) / hectares
    else:
        prediction_ton_per_ha = 0

    st.subheader("Hasil Prediksi")
    st.write(f"Total Produksi: {prediction_kg:.2f} kg")
    st.write(f"Predicted harvest yield: {prediction_ton_per_ha:.2f} ton/hektar")

    st.subheader("Penjelasan Prediksi")
    if prediction_ton_per_ha > 5.5:
        st.write("Prediksi tinggi. Kombinasi pupuk urea & potash yang cukup, ditambah curah hujan dan kelembaban yang mendukung berkontribusi positif terhadap hasil panen.")
    elif prediction_ton_per_ha > 4.0:
        st.write("Prediksi sedang. Hasil masih cukup baik, tapi bisa ditingkatkan dengan penyesuaian pupuk atau pemantauan kelembaban lebih lanjut.")
    else:
        st.write("Prediksi rendah. Perhatikan kembali jumlah pupuk dan kondisi curah hujan/kelembaban di 30 hari pertama.")

    st.caption("Planting method dan temperature ditampilkan sebagai input tambahan, namun belum digunakan dalam model karena keterbatasan dataset.")

else:
    st.info("Masukkan data di sidebar lalu klik tombol **Prediksi Hasil Panen**")
