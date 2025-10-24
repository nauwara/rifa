import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ==========================
# KONFIGURASI DASBOR
# ==========================
st.set_page_config(
    page_title="Dashboard Rifa Nauwara",
    page_icon="🎓",
    layout="wide"
)

# ==========================
# GAYA (TEMA PASTEL)
# ==========================
st.markdown(
    """
    <style>
        /* Warna latar pastel */
        .stApp {
            background-color: #FFF7F0;
            color: #4A3F35;
        }

        /* Judul utama */
        h1, h2, h3 {
            color: #4A3F35;
            font-weight: 800;
        }

        /* Paragraf dan teks umum */
        p, label, span, div {
            color: #5A4E44;
        }

        /* Tombol */
        .stButton>button {
            background-color: #F9D5E5;
            color: #4A3F35;
            border-radius: 12px;
            border: none;
            font-weight: 600;
            transition: 0.3s;
        }

        .stButton>button:hover {
            background-color: #E5BACE;
            color: white;
        }

        /* Kotak upload */
        [data-testid="stFileUploader"] {
            background-color: #FFF0F5;
            border: 2px dashed #E5BACE;
            border-radius: 15px;
            padding: 15px;
        }

        /* Garis pembatas */
        hr {
            border: 1px solid #E5BACE;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# HEADER
# ==========================
st.markdown("<h1>🎓 FELIDAE SPECIES CLASSIFICATION DASHBOARD </h1>", unsafe_allow_html=True)
st.markdown("### 🐯 LAPORAN 5 PRAKTIKUM BIG DATA RIFA NAUWARA")
st.caption("Praktikum Big Data — Universitas Syiah Kuala")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_classifier():
    model = load_model("model/RifaNauwara_Laporan2.h5")  # pastikan path dan nama file sesuai
    return model

classifier = load_classifier()

# ==========================
# LABEL KELAS
# ==========================
classes = ["Cheetah", "Leopard", "Lion", "Puma", "Tiger"]

species_info = {
    "Cheetah": "🐆 Cheetah dikenal sebagai hewan darat tercepat di dunia, dengan tubuh ramping dan bercak hitam khas.",
    "Leopard": "🐆 Leopard memiliki kemampuan memanjat pohon yang hebat dan pola tutul roset di seluruh tubuhnya.",
    "Lion": "🦁 Lion atau singa dikenal sebagai 'Raja Hutan', biasanya hidup berkelompok (pride).",
    "Puma": "🐈‍⬛ Puma atau cougar memiliki tubuh kuat dan bulu berwarna coklat keabu-abuan.",
    "Tiger": "🐯 Tiger adalah kucing terbesar di dunia, mudah dikenali dari garis-garis hitam di tubuhnya."
}

# ==========================
# BAGIAN UNGGAH GAMBAR
# ==========================
st.markdown("---")
st.subheader("📸 Unggah dan Klasifikasikan Gambar Felidae")
uploaded_file = st.file_uploader("Unggah gambar spesies kucing besar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="📷 Gambar yang diunggah", use_container_width=True)

    try:
        input_shape = classifier.input_shape[1:3]
        st.caption(f"📏 Ukuran input model: {input_shape}")

        img_resized = img.resize(input_shape)
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        with st.spinner("🔍 Menganalisis gambar..."):
            prediction = classifier.predict(img_array)
            class_index = np.argmax(prediction)
            predicted_label = classes[class_index]
            confidence = np.max(prediction)

        # Hasil
        st.success(f"✅ Prediksi: **{predicted_label}**")
        st.metric("Tingkat Keyakinan Model", f"{confidence*100:.2f}%")

        st.markdown("---")
        st.markdown(f"### Tentang {predicted_label}")
        st.write(species_info[predicted_label])

        # Grafik
        st.markdown("### 📊 Probabilitas Tiap Kelas")
        fig, ax = plt.subplots()
        ax.bar(classes, prediction[0], color=['#F4A261', '#2A9D8F', '#E76F51', '#264653', '#E9C46A'])
        ax.set_ylabel("Probabilitas")
        ax.set_xlabel("Kelas")
        ax.set_title("Distribusi Keyakinan Model")
        st.pyplot(fig)

        # Tombol ulang
        st.markdown("---")
        if st.button("🔁 Coba Gambar Lain"):
            st.experimental_rerun()

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat memproses gambar.")
        st.caption(f"Detail error: {str(e)}")

else:
    st.info("📂 Silakan unggah gambar untuk mulai klasifikasi.")

# ==========================
# STATISTIK MODEL
# ==========================
st.markdown("---")
st.subheader("📈 Statistik Model Felidae")

col1, col2 = st.columns(2)
with col1:
    st.metric("Jumlah Dataset", "243 gambar")
    st.metric("Jumlah Kelas", "5 spesies kucing besar")
with col2:
    st.metric("Akurasi Training", "92.7%")
    st.metric("Akurasi Validasi", "88.3%")

st.caption("📊 Data diambil dari hasil pelatihan model CNN Felidae — Rifa Nauwara")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("Dibuat oleh **Rifa Nauwara** | 💕 Praktikum Big Data — 2025")
