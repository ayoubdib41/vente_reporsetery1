
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 📦 Chargement des objets
model_sales = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# 🎯 Configuration de l'application
st.set_page_config(page_title="Prédiction des ventes", page_icon="📊")
st.title("📊 Application de Prédiction des ventes et quantités")

# 💡 Partie 1 : Choix du type de prédiction temporelle
st.header("🧠 Partie 1 : Choix du type de prédiction temporelle")
granularite = st.selectbox("Niveau de granularité temporelle :", ["Jour complet", "Année + Mois", "Année"])
annee = st.selectbox("Année", list(range(2015, 2025)))
mois = st.slider("Mois", 1, 12, 6)
jour_semaine = st.slider("Jour de semaine (0=Lundi)", 0, 6, 0)

# 📦 Partie 2 : Type de produit
st.header("📦 Partie 2 : Type de produit")
type_produit = st.radio("Filtrer les produits par :", ["Tous les produits", "Par catégorie", "Par sous-catégorie"])

# 🔧 Partie 3 : Informations complémentaires
st.header("📌 Informations complémentaires")
is_holiday = st.selectbox("Jour férié ?", ["Non", "Oui"])
is_holiday_season = st.selectbox("Saison des fêtes ?", ["Non", "Oui"])
delivery_duration = st.number_input("Durée de livraison (jours)", min_value=0)

# Création du dictionnaire d'entrée avec les features attendues
input_data = {}
for feature in features:
    if feature in [
        "Order_Year", "Order_Month", "Order_DayOfWeek",
        "Is_Holiday", "Is_Holiday_Season", "Delivery_Duration"
    ]:
        continue  # ces champs seront ajoutés manuellement
    input_data[feature] = 0.0  # valeur neutre par défaut

input_data.update({
    "Order_Year": annee,
    "Order_Month": mois,
    "Order_DayOfWeek": jour_semaine,
    "Is_Holiday": 1 if is_holiday == "Oui" else 0,
    "Is_Holiday_Season": 1 if is_holiday_season == "Oui" else 0,
    "Delivery_Duration": delivery_duration
})

# 🧠 Lancement de la prédiction
if st.button("🧾 Prédire les Ventes et Quantités"):
    try:
        df_input = pd.DataFrame([input_data])
        X_scaled = scaler.transform(df_input)
        y_pred = model_sales.predict(X_scaled)[0]

        st.success("✅ Prédictions réussies !")
        st.markdown("### Résultats de la prédiction :")
        st.markdown(f"- **Sales prédit** : 💰 **{y_pred:,.2f} €**")
        st.markdown(f"- **Quantity prédit** : 📦 **{int(input_data['Quantity'])}**")
    except Exception as e:
        st.error(f"⚠️ Erreur : {e}")
