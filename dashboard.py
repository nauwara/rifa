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
    page_title="Felidae Classifier Dashboard",
    page_icon="🐯",
    layout="wide"
)

st.title("🐯 Felidae Species Classification Dashboard")
st.markdown("### UTS Lab Big Data — Rifa Nauwara")

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

# Deskripsi singkat tiap kelas
species_info = {
    "Cheetah": "🐆 Cheetah dikenal sebagai hewan darat tercepat di dunia, dengan tubuh ramping dan bercak hitam khas.",
    "Leopard": "🐆 Leopard memiliki kemampuan memanjat pohon yang hebat dan pola tutul roset di seluruh tubuhnya.",
    "Lion": "🦁 Lion atau singa dikenal sebagai 'Raja Hutan', biasanya hidup berkelompok (pride).",
    "Puma": "🐈‍⬛ Puma atau cougar memiliki tubuh kuat dan bulu berwarna coklat keabu-abuan.",
    "Tiger": "🐯 Tiger adalah kucing terbesar di dunia, mudah dikenali dari garis-garis hitam di tubuhnya."
}

# ==========================
# TABS UNTUK MULTI-MODEL (FELIDAE & DIGITS)
# ==========================
tab1, tab2 = st.tabs(["🐯 Felidae Classifier"])

# ==========================
# TAB 1 — FELIDAE CLASSIFIER
# ==========================
with tab1:
    uploaded_file = st.file_uploader("Unggah gambar spesies kucing besar:", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

        # ==========================
        # PREPROCESSING
        # ==========================
        try:
            input_shape = classifier.input_shape[1:3]
            st.caption(f"📏 Ukuran input model: {input_shape}")

            img_resized = img.resize(input_shape)
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0)

            if np.max(img_array) > 1:
                img_array = img_array / 255.0

            # ==========================
            # PREDIKSI
            # ==========================
            with st.spinner("🔍 Menganalisis gambar..."):
                prediction = classifier.predict(img_array)
                class_index = np.argmax(prediction)
                predicted_label = classes[class_index]
                confidence = np.max(prediction)

            # ==========================
            # HASIL PREDIKSI
            # ==========================
            st.success(f"✅ Prediksi: **{predicted_label}**")
            st.metric("Tingkat Keyakinan Model", f"{confidence*100:.2f}%")

            st.markdown("---")
            st.markdown(f"### Tentang {predicted_label}")
            st.write(species_info[predicted_label])

            # ==========================
            # VISUALISASI PROBABILITAS
            # ==========================
            st.markdown("### 📊 Probabilitas Tiap Kelas")
            fig, ax = plt.subplots()
            ax.bar(classes, prediction[0], color=['#F4A261', '#2A9D8F', '#E76F51', '#264653', '#E9C46A'])
            ax.set_ylabel("Probabilitas")
            ax.set_xlabel("Kelas")
            ax.set_title("Distribusi Keyakinan Model")
            st.pyplot(fig)

            # ==========================
            # TOMBOL ULANG
            # ==========================
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
# TAB 2 — DIGIT CLASSIFIER (Opsional)
# ==========================
with tab2:
    st.info("🔢 Mode klasifikasi digit belum diaktifkan.")
    st.caption("Nanti kamu bisa tambahkan model .pt untuk mendeteksi angka dan menentukan apakah ganjil/genap.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("Dibuat oleh **Rifa Nauwara** | UTS Lab Big Data 🧠")
