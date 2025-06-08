
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🎯 Configuration de la page
st.set_page_config(page_title="Prédiction des ventes", page_icon="📊", layout="centered")

# 📦 Chargement des modèles et outils
model_sales = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")
features = joblib.load("features.pkl")

# 🧠 Partie 1 : Choix temporel
st.title("📊 Application de Prédiction des ventes et quantités")
st.header("🧠 Partie 1 : Choix du type de prédiction temporelle")
granularite = st.selectbox("Niveau de granularité temporelle :", ["Jour complet", "Année + Mois", "Année"])
annee = st.selectbox("Année", list(range(2015, 2021)))
mois = st.slider("Mois", 1, 12, 6)
semaine = st.slider("Semaine", 1, 52, 26)
jour_semaine = st.slider("Jour de semaine (0=Lundi)", 0, 6, 0)

# 📦 Partie 2 : Produit
st.header("📦 Partie 2 : Type de produit")
type_produit = st.radio("Filtrer les produits par :", ["Tous les produits", "Par catégorie", "Par sous-catégorie"])

# 📌 Partie 3 : Infos complémentaires
st.header("📌 Informations complémentaires")
is_holiday = st.selectbox("Jour férié ?", ["Non", "Oui"])
is_holiday_season = st.selectbox("Saison des fêtes ?", ["Non", "Oui"])
delivery_duration = st.number_input("Durée de livraison (jours)", min_value=0)

# 🧮 Partie 4 : Variables du modèle
st.subheader("📝 Remplir les caractéristiques du scénario :")
input_data = {}
for feature in features:
    if feature not in ["Order_Year", "Order_Month", "Order_Week", "Order_DayOfWeek", "Is_Holiday", "Is_Holiday_Season", "Delivery_Duration"]:
        input_data[feature] = st.number_input(f"{feature}", step=0.01)

# Ajout des variables calculées
input_data.update({
    "Order_Year": annee,
    "Order_Month": mois,
    "Order_Week": semaine,
    "Order_DayOfWeek": jour_semaine,
    "Is_Holiday": 1 if is_holiday == "Oui" else 0,
    "Is_Holiday_Season": 1 if is_holiday_season == "Oui" else 0,
    "Delivery_Duration": delivery_duration
})

# 🎯 Lancer la prédiction
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
