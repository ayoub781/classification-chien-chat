import base64
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
# --- Chargement du fond d'écran local ---
def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(255, 255, 255, 0.8);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- Appliquer le fond ---
set_background("chienvschat.jpeg")
# --- Configuration de la page ---
st.set_page_config(page_title="Classificateur Chien/Chat", page_icon="🐾", layout="centered")
# --- Styles personnalisés pour le texte ---
st.markdown("""
<style>
h1 {
    font-size: 3em !important;
    font-weight: 800 !important;
    text-align: center;
    color: white !important;
    text-shadow: 2px 2px 6px rgba(0,0,0,0.7);
}
h3 {
    font-size: 1.5em !important;
    font-weight: 600 !important;
    text-align: center;
    color: #f0f0f0 !important;
    text-shadow: 1px 1px 4px rgba(0,0,0,0.6);
}
div.block-container {
    background: rgba(0, 0, 0, 0.35);
    border-radius: 20px;
    padding: 30px;
    backdrop-filter: blur(8px);
}
</style>
""", unsafe_allow_html=True)
# --- Titre et description ---
st.title("Classification Chien vs Chat")
st.markdown("### Téléverse une image pour savoir si c’est un **chien** ou un **chat**")

# --- Chargement du modèle ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model_chien_chat.keras")  # ou .h5 selon ton choix
    return model

model = load_model()

# --- Upload de l'image ---
uploaded_file = st.file_uploader("Choisis une image 🖼️", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture de l’image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Image importée", use_container_width=True)

    # Prétraitement de l'image pour le modèle
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prédiction
    prediction = model.predict(img_array)
    prob = prediction[0][0]

    # --- Résultat ---
    st.markdown("---")
    if prob > 0.5:
        st.success(f"🐶 C’est très probablement un **chien** ! (confiance : {prob*100:.2f}%)")
    else:
        st.info(f"🐱 C’est très probablement un **chat** ! (confiance : {(1-prob)*100:.2f}%)")

    st.markdown("---")
    st.caption("Modèle CNN entraîné avec TensorFlow et Keras 🧠")