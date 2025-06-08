
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Chargement des objets nécessaires
model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# 🧾 Titre de l'application
st.title("📈 Prédiction des ventes - Superstore")

# 🕒 Choix de la granularité temporelle
granularite = st.selectbox(
    "Choisir la granularité de la prédiction :",
    ["Année", "Année + Mois", "Jour"]
)

# 📦 Choix du niveau de segmentation
segmentation = st.selectbox(
    "Prédire pour :",
    ["Tous les produits", "Par Catégorie", "Par Sous-Catégorie"]
)

# 🧮 Formulaire d'entrée
st.subheader("📝 Remplir les caractéristiques du scénario :")
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(f"{feature}", value=0.0)

# 🔍 Lancement de la prédiction
if st.button("🔮 Prédire les ventes"):
    try:
        df_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df_input)
        prediction = model.predict(X_scaled)[0]
        prediction = np.expm1(prediction) if "Log" in features[0] else prediction
        st.success(f"✅ Prédiction : {prediction:,.2f} $")
    except Exception as e:
        st.error(f"⚠️ Erreur lors de la prédiction : {e}")
