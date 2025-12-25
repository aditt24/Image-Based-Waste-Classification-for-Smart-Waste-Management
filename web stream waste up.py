import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. CONFIG
st.set_page_config(page_title="Waste Classification", page_icon="♻️", layout="centered")

CLASS_NAMES = [
    "Compost (Organik)",       
    "Unecyclable (Residu)", 
    "Recyclable (Daur Ulang)"  
]

IMG_SIZE = 224

# 2. LOAD MODEL 
@st.cache_resource
def load_model():
    model_path = r"C:\Users\LENOVO\OneDrive\Documents\Program\Waste clasifikasi\waste_classifier.keras"
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model()

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(image).astype(np.float32) 
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# 4. STATE MANAGEMENT
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []

def pindah_ke_home():
    st.session_state.page = 'home'
    st.session_state.uploaded_files_list = []
    st.rerun()

# --- HALAMAN 1: HOME ---
if st.session_state.page == 'home':
    st.title("Waste Classification App")

    waste_image_path = r"C:\Users\LENOVO\OneDrive\Documents\Program\Waste clasifikasi\waste.jpg"
    try:
        st.image(waste_image_path, caption="Sistem Klasifikasi Sampah", use_container_width=True)
    except:
        pass

    st.markdown("""
    ### Kategori Sampah
    ** Recyclable (Daur Ulang)** Sampah yang masih dapat didaur ulang seperti botol plastik, kaleng, kaca, dan kertas.            
    ** Compost (Organik)** Sampah yang mudah terurai secara alami seperti sisa makanan, daun, dan limbah dapur.     
    ** Unecyclable (Residu)** Sampah yang tidak dapat didaur ulang maupun dikomposkan seperti baterai, tisu kotor, dan styrofoam.
    """)

    st.markdown("---")
    uploaded_files = st.file_uploader("Pilih satu atau beberapa gambar...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if st.button("Prediksi", use_container_width=True):
            st.session_state.uploaded_files_list = uploaded_files
            st.session_state.page = 'result'
            st.rerun()

# --- HALAMAN 2: RESULT ---
elif st.session_state.page == 'result':
    st.title("Hasil Klasifikasi")
    st.markdown("---")
    
    if not st.session_state.uploaded_files_list:
        st.warning("Tidak ada gambar.")
        if st.button("Kembali"): pindah_ke_home()
    else:
        with st.spinner("Menganalisis..."):
            for i, file in enumerate(st.session_state.uploaded_files_list):
                image = Image.open(file)
                input_tensor = preprocess_image(image)
                
                if model is not None:
                    preds = model.predict(input_tensor, verbose=0)[0]
                    class_id = np.argmax(preds)
                    result_label = CLASS_NAMES[class_id]
                    confidence = np.max(preds) * 100
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.image(image, caption=f"Gambar {i+1}", use_container_width=True)
                    with col2:
                        st.success(f"### Gambar {i+1}:\n## {result_label}")
                        st.write(f"**Akurasi:** {confidence:.2f}%")
                        st.info("Saran: Buanglah ke tempat sampah yang sesuai.")
                    st.markdown("---")
                else:
                    st.error("Model tidak tersedia.")

        if st.button(" Kembali ke Home", use_container_width=True):
            pindah_ke_home()

# streamlit run "C:\Users\LENOVO\OneDrive\Documents\Program\Waste clasifikasi\web stream waste up.py"