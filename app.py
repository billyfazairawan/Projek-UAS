import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Memuat kedua model
densenet_model = load_model('dense_model.h5')
nasnet_model = load_model('nasnet_mobile.h5')

# Nama kelas penyakit
class_names = ['Bercak Daun', 'Daun Sehat', 'Hawar Daun', 'Karat Daun']

# Fungsi untuk memprediksi penyakit dari gambar
def predict_disease(model, img):
    img = img.resize((224, 224))  # Resize sesuai input model
    img = np.array(img.convert('RGB'))  # Pastikan format RGB
    img = img / 255.0  # Normalisasi
    img = np.expand_dims(img, axis=0)  # Tambah batch dimensi
    pred = model.predict(img)
    return np.argmax(pred, axis=1)[0], np.max(pred)

# Antarmuka Streamlit
st.title("Perbandingan Model: Prediksi Penyakit Daun Jagung")
st.write("Unggah gambar daun dan pilih model untuk melihat prediksi penyakitnya.")

# Input gambar
uploaded_file = st.file_uploader("Pilih Gambar Daun Jagung", type=["jpg", "png", "jpeg"])

# Pilihan model
model_choice = st.selectbox("Pilih Model untuk Prediksi", ["DenseNet", "NASNet"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diunggah", use_container_width=True)

    if st.button("Prediksi Penyakit"):
        # Gunakan model sesuai pilihan
        if model_choice == "DenseNet":
            label_idx, confidence = predict_disease(densenet_model, img)
        else:
            label_idx, confidence = predict_disease(nasnet_model, img)

        label = class_names[label_idx]

        st.subheader(f"Hasil Prediksi dengan {model_choice}:")
        st.write(f"Penyakit yang terdeteksi: **{label}**")
        st.write(f"Tingkat Kepercayaan: **{confidence * 100:.2f}%**")

        # Penjelasan tambahan
        if label != "Daun Sehat":
            st.warning(f"Daun ini terdeteksi mengidap **{label}**. Segera lakukan penanganan!")
        else:
            st.success("Daun ini sehat, tidak terdeteksi penyakit.")
