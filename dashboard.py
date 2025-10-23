import streamlit as st
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# ==========================
# KONFIGURASI DASBOR
# ==========================
st.set_page_config(
    page_title="Felidae Classifier Dashboard",
    page_icon="🐯",
    layout="wide"
)

st.title("🐯 Felidae Species Classification Dashboard")
st.markdown("### UTS Big Data — Rifa Nauwara")

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_classifier():
    model = load_model("model/RifaNauwara_Laporan2.h5")  # pastikan nama dan path sesuai di repo GitHub kamu
    return model

classifier = load_classifier()

# Ambil ukuran input otomatis dari model
input_shape = classifier.input_shape[1:3]  # contoh: (128, 128)
st.caption(f"📏 Ukuran input model: {input_shape}")

# ==========================
# LABEL KELAS
# ==========================
classes = ["Cheetah", "Leopard", "Lion", "Puma", "Tiger"]

# deskripsi tiap kelas
species_info = {
    "Cheetah": "🐆 Cheetah dikenal sebagai hewan darat tercepat di dunia, dengan tubuh ramping dan bercak hitam khas.",
    "Leopard": "🐆 Leopard memiliki kemampuan memanjat pohon yang hebat dan pola tutul roset di seluruh tubuhnya.",
    "Lion": "🦁 Lion atau singa dikenal sebagai 'Raja Hutan', biasanya hidup berkelompok (pride).",
    "Puma": "🐈‍⬛ Puma atau cougar memiliki tubuh kuat dan bulu berwarna coklat keabu-abuan.",
    "Tiger": "🐯 Tiger adalah kucing terbesar di dunia, mudah dikenali dari garis-garis hitam di tubuhnya."
}

# ==========================
# UNGGAH GAMBAR
# ==========================
uploaded_file = st.file_uploader("Unggah gambar spesies kucing besar:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Gambar yang diunggah", use_container_width=True)

    # ==========================
    # PREPROCESSING
    # ==========================
    img_resized = img.resize(input_shape)  # otomatis menyesuaikan ukuran model
    img_array = image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
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

else:
    st.info("📂 Silakan unggah gambar untuk mulai klasifikasi.")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.caption("Dibuat oleh **Rifa Nauwara** | UTS Lab Big Data 🧠")
