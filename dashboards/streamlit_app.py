import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image
import os

# Chargement du logo (adapté pour Heroku)
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")

if os.path.exists(logo_path):
    logo = Image.open(logo_path)
    st.image(logo, width=150) 
st.markdown("""
Bienvenue sur l'application de scoring de risque de défaut.  
Cette application permet de prédire la probabilité qu'un client ne rembourse pas son crédit,  
en se basant sur un modèle de machine learning entraîné sur des données réelles.

Saisissez un identifiant client pour obtenir la prédiction et des explications visuelles avec SHAP.
""")

# Config API - modifier selon local ou cloud
API_URL = st.secrets.get("API_URL") or "http://localhost:8000"

st.title("Prédiction risque défaut - Interface")

client_id = st.number_input("Saisir un ID client", min_value=1, step=1)

if st.button("Prédire le risque"):
    response = requests.get(f"{API_URL}/predict/{client_id}")
    if response.status_code == 200:
        data = response.json()
        if "error" in data:
            st.error(data["error"])
        else:
            st.write("Résultat prédiction :")
            st.json(data)
    else:
        st.error(f"Erreur API : {response.status_code}")

# --- Visualisation SHAP Global ---
st.subheader("SHAP - Importance Globale des Features")

if st.button("Afficher l'explication globale (SHAP)"):
    response = requests.get(f"{API_URL}/shap/global")
    if response.status_code == 200:
        shap_data = response.json()
        features = shap_data["features"]
        values = shap_data["values"]

        shap_df = pd.DataFrame({"Feature": features, "Importance": values})
        shap_df = shap_df.sort_values("Importance", ascending=True)

        fig, ax = plt.subplots()
        ax.barh(shap_df["Feature"], shap_df["Importance"])
        ax.set_title("Importance des variables (SHAP global)")
        st.pyplot(fig)
    else:
        st.error("Erreur lors de la récupération des SHAP global.")

# --- Visualisation SHAP Local ---
st.subheader("SHAP - Explication Locale pour ce client")

if st.button("Afficher l'explication locale (SHAP)"):
    response = requests.get(f"{API_URL}/shap/local/{client_id}")
    if response.status_code == 200:
        shap_data = response.json()
        shap_values = shap_data["shap_values"]
        expected_value = shap_data["expected_value"]
        features = shap_data["features"]

        shap_df = pd.DataFrame([features])
        explainer = shap.Explanation(values=shap_values,
                                     base_values=expected_value,
                                     data=shap_df,
                                     feature_names=list(features.keys()))

        st.set_option("deprecation.showPyplotGlobalUse", False)
        shap.plots.waterfall(explainer[0])
        st.pyplot(bbox_inches='tight')
    else:
        st.error("Erreur lors de la récupération des SHAP local.")