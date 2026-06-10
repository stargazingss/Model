import streamlit as st
import pandas as pd
import joblib

from PIL import Image

@st.cache_resource
def load_model():
    model = joblib.load("best_random_forest_model.pkl")
    scaler = joblib.load("robust_standard_scaler.pkl")
    columns = joblib.load("feature_columns.pkl")

    return model, scaler, columns, 

model, scaler, columns = load_model()

st.set_page_config(
    page_title="Paddy Yield Prediction",
    page_icon="🌾",
    layout="wide"
)

img = Image.open("logo_padi.jpg")

# Ambil bagian tengah gambar
width, height = img.size
img = img.crop((((0, 150, 735, 400))))

st.image(img, use_container_width=True)

st.title("🌾 Paddy Yield Prediction System")
st.write(
    """
    Sistem ini digunakan untuk memperkirakan hasil panen padi berdasarkan
    kondisi lahan, penggunaan benih, pemupukan, serta perlindungan tanaman.
    """
)

st.header("🧱Luas Lahan Utama (Hectares)")

st.caption(
    "Masukkan total luas lahan sawah aktif yang digunakan "
    "untuk budidaya padi pada musim tanam saat ini."
)

st.markdown("**Luas Lahan (ha)**")
st.caption("Luas total area sawah aktif yang Anda tanami padi saat ini.")

hectares = st.number_input(
    label="",
    min_value=0.0,
    value=2.5,
    label_visibility="collapsed"
)

st.markdown("**Jumlah Benih Padi(kg)**")
st.caption("Berat total benih yang disemai untuk kebutuhan musim tanah ini.")

seedrate = st.number_input(
    label="",
    min_value=0.0,
    value=150.0,
    help="Berapa total bobot benih padi yang Anda sebar atau semai untuk musim tanam ini?"
)

st.header("🌱Area Pembibitan Awal (Nursery)")

st.caption(
    "Masukkan informasi mengenai area pembibitan dan persiapan tanah yang digunakan. "
    "Untuk menumbuhkan bibit padi sebelum dipindahkan ke lahan utama"
)

st.markdown("**Luas Area Pembibitan(Cents)**")
st.caption("Luas petak tanah khusus yang Anda gunakan untuk menyemai benih awal sebelum nantinya dipindahkan ke sawah utama.")

nursery_area = st.number_input(
    label="",
    min_value=0.0,
    value=120.0,
    help="Masukkan luas petak khusus yang Anda gunakan untuk menyemai bibit awal sebelum dipindah ke sawah utama."
)

st.markdown("**Persiapan Tanah Pembibitan (Ton)**")
st.caption("Banyaknya pupuk organik atau kompos yang diberikan untuk mengelolah dan menggemburkan tanah khusus di area pembibitan awal.")

lp_nursery = st.number_input(
    label="",
    min_value=0.0,
    value=6.0,
    help="Berapa banyak pupuk organik atau kompos yang Anda gunakan untuk mengolah tanah di tempat pembibitan awal?"
)

st.header("🚜Pengolahan Lahan Utama")

st.caption(
    "Masukkan data terkait lahan sawah utama, termasuk penggunaan bahan organik, "
    "dan pengelolaan sisa tanaman yang dapat memengaruhi kesuburan tanah"
)

st.markdown("**Persiapan Lahan Utama (Ton)**")
st.caption("Total berat pupuk organikm, kompos, atau kapur dasar yang diatur saat pertama kali membajak sawah utama untuk menyiapkan kondisi tanah pra-tanam.")

lp_mainfield = st.number_input(
    label="",
    min_value=0.0,
    value=75.0,
    help="Berapa banyak total pupuk organik atau kompos yang ditabur saat pertama kali membajak sawah utama sebelum ditanami?"
)

st.markdown("**Pengelolaan Jerami/Sisa Sawah (Bundles)**")
st.caption("Jumlah ikatan jerami atau sisa rumput kering hasil panen lalu yang dihamparkan kembali atau dibenamkan ke dalam tanah sawah sebagai mulsa alami.")

trash = st.number_input(
    label="",
    min_value=0.0,
    value=540.0,
    help="Berapa banyak ikatan jerami atau sisa rumput kering yang Anda hamparkan kembali ke sawah sebagai penutup tanah alami?"
)

st.header("🧪Pemupukan & Nutrisi Tanaman")

st.caption(
    "Masukkan jumlah pupuk dan nutrisi yang diberikan pada berbagai fase pertumbuhan padi. "
    "Informasi ini digunakan untuk memperkirakan pengaruh penumpukan terhadap hasil panen"
)

st.markdown("**Pupuk DAP Hari Ke-20 (Kg)**")
st.caption("Dosis pupuk Di-ammonium Phosphate (DAP/Unsur fosfor) yang diberikan saat usia padi menginjak 20 hari setelah tanam untuk merangsang pertumbuhan akar.")

dap = st.number_input(
    label="",
    min_value=0.0,
    value=240.0,
    help="Masukkan jumlah pupuk DAP yang Anda tabur saat usia padi menginjak 20 hari setelah tanam."
)

st.markdown("**Pupuk Urea Hari Ke-40**")
st.caption("Dosis pupuk Nitrogen (Urea) yang ditaburkan pada umur tanaman 40 hari guna memacu pertumbuhan hijau daun dan batang 9fase vegetatif")

urea = st.number_input(
    label="",
    min_value=0.0,
    value=162.78,
    help="Masukkan total pupuk Urea yang diberikan pada fase pertumbuhan daun di hari ke-40."
)

st.markdown("**Pupuk Mikro Hari Ke-50**")
st.caption("Dosis pupuk Kalium (Potash/KCL) yang ditaburkan pada umur tanaman 50 hari untuk membantu kesiapan padi dalam proses pengisian bulir.")

potash = st.number_input(
    label="",
    min_value=0.0,
    value=62.28,
    help="Masukkan jumlah pupuk Kalium (Potash) yang diberikan saat tanaman padi mulai mempersiapkan pembuahan di hari ke-50."
)

st.markdown("**Pupuk Mikro Hari Ke-70**")
st.caption("Dosis suplemen nutrisi tambahan atau zat hara mikro (seperti seng besi, mangan) yang disemprotkan pada umur 70 hari saat tanaman memasuki fase pembentukan buir padi")

micronutrients = st.number_input(
    label="",
    min_value=0.0,
    value=90.0,
    help="Masukkan jumlah suplemen nutrisi tambahan atau vitamin padi yang Anda semprotkan pada fase pengisian bulir di hari ke-70."
)

st.header("🐛Perlindungan Tanaman")

st.caption(
    "Masukkan data penggunaan herbisida dan pestisida yang diterapkan selama masa tanam. "
    "Untuk melindungi tanaman dari gulma, hama, dan penyakit"
)

st.markdown("**Herbisida Hari Ke-28**")
st.caption("Dosis cairan obat pembasmi rumput liar/gulma (khususnya yang berbahan kimia Thiobencarb) yang digunakan untuk membersihkan area sawah pada hari ke-28.")

weed = st.number_input(
    label="",
    min_value=0.0,
    value=12.0,
    help="Berapa dosis obat pembasmi gulma (Thiobencarb) yang digunakan untuk membersihkan rumput liar di sekitar padi pada hari ke-28?"
)

st.markdown("**Peptisida Hari Ke-60**")
st.caption("Volume  cairan pembasmi serangga (peptisida/insektisida/fungisida) yang disemprotkan pada umur 60 hari untuk membentengi tanaman dari serangan hama penyakit.")

pest = st.number_input(
    label="",
    min_value=0.0,
    value=3600.0,
    help="Masukkan total dosis cairan pembasmi serangga atau hama yang disemprotkan untuk melindungi padi pada hari ke-60."
)

if st.button("🌾 Prediksi Hasil Panen"):

    st.markdown("""
    Aplikasi ini digunakan untuk memperkirakan hasil panen padi berdasarkan kondisi lahan,
    penggunaan benih, pemupukan, serta perlindungan tanaman selama masa budidaya.

    Silakan isi seluruh data sesuai kondisi aktual di lapangan untuk mendapatkan hasil prediksi yang lebih akurat.
    """)

    input_df = pd.DataFrame({
        'Hectares ': [hectares],
        'Micronutrients_70Days': [micronutrients],
        'Potassh_50Days': [potash],
        'Urea_40Days': [urea],
        'Pest_60Day(in ml)': [pest],
        'LP_Mainfield(in Tonnes)': [lp_mainfield],
        'DAP_20days': [dap],
        'Trash(in bundles)': [trash],
        'Seedrate(in Kg)': [seedrate],
        'LP_nurseryarea(in Tonnes)': [lp_nursery],
        'Weed28D_thiobencarb': [weed],
        'Nursery area (Cents)': [nursery_area]
    })

    try:
        input_df = input_df.reindex(columns=columns, fill_value=0)

        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]

        st.success(
            f"Perkiraan Hasil Panen Padi: {prediction:,.2f} Kg"
        )

        st.subheader("Interpretasi Prediksi")

        explanation = []

        if hectares > 2:
            explanation.append("Luas lahan yang lebih besar berkontribusi terhadap peningkatan hasil panen.")

        if urea > 150:
            explanation.append("Pemberian pupuk Urea yang cukup mendukung pertumbuhan vegetatif tanaman.")

        if potash > 50:
            explanation.append("Pupuk Kalium membantu proses pembentukan dan pengisian bulir padi.")

        if micronutrients > 70:
            explanation.append("Nutrisi mikro mendukung perkembangan tanaman pada fase generatif.")

        if not explanation:
            explanation.append("Prediksi dihitung berdasarkan kombinasi seluruh faktor budidaya yang dimasukkan.")

        for item in explanation:
            st.write("•", item)

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")

st.markdown("---")

st.info(
    """
    Catatan:
    - Gunakan tanda titik untuk angka desimal (contoh: 62.28).
    - Jika tidak memiliki data pasti, gunakan estimasi yang mendekati kondisi lapangan.
    """
)