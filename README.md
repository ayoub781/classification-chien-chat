# 🐶 Classification Chien vs Chat 🐱  
*(CNN + Transfer Learning avec TensorFlow & Keras)*  

## 🇫🇷 Description du projet

Ce projet est une **application Streamlit** permettant de classifier une image comme étant un **chien** 🐕 ou un **chat** 🐈.  
Le modèle a été entraîné à l’aide du **transfer learning** (réutilisation d’un modèle pré-entraîné sur ImageNet, puis fine-tuning).  

L’application permet à l’utilisateur de :
- Téléverser une image (.jpg, .jpeg, .png)  
- Obtenir une prédiction immédiate ("Chien", "Chat" ou "Incertitude")  
- Visualiser la confiance du modèle  

> ⚠️ Ce modèle est spécifiquement entraîné pour distinguer **les chiens et les chats**.  

---

## 🧠 Modèle utilisé

- Architecture : **MobileNetV2** (transfer learning)
- Framework : **TensorFlow / Keras**
- Entraînement : 10 époques, accuracy ≈ **0.98**
- Fichier du modèle : `model_chien_chat_transfer.h5`

---

## 🚀 Démo en ligne

👉 [**Accéder à l’application Streamlit**](https://classification-chien-chat-ayoub.streamlit.app)

---

