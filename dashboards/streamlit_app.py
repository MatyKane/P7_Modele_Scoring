import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np

st.title("Scoring Model Interface")

st.markdown("""
Bienvenue sur l'application de scoring de risque de défaut.  
Cette application permet de prédire la probabilité qu'un client ne rembourse pas son crédit,  
en se basant sur un modèle de machine learning entraîné sur des données réelles.

Saisissez un identifiant client pour obtenir la prédiction et des explications visuelles avec SHAP.
""")

# Config API - modifier selon local ou cloud
API_URL = st.secrets.get("API_URL") or "https://solvability.onrender.com"

# Barre latérale navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisissez une section :", ["Accueil", "Prédiction", "SHAP Global", "SHAP Local"])

# Input client_id commun à toutes les sections (on évite de demander plusieurs fois)
client_id = st.number_input("Saisir un ID client", min_value=1, step=1, key="client_id_input")

if section == "Accueil":
    st.write("Sélectionnez une section dans la barre latérale pour commencer.")

elif section == "Prédiction":
    seuil = st.slider("Seuil métier", min_value=0.0, max_value=1.0, value=0.545, step=0.01)

    if st.button("Prédire le risque"):
        try:
            url = f"{API_URL}/predict/{client_id}?seuil={seuil}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.subheader("Résultat prédiction")
                st.json(data)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête API : {e}")

elif section == "SHAP Global":
    if st.button("Afficher l'explication globale (SHAP)"):
        try:
            response = requests.get(f"{API_URL}/shap/global")
            response.raise_for_status()
            shap_data = response.json()

            features = shap_data["features"]
            values = shap_data["values"]

            shap_df = pd.DataFrame({"Feature": features, "Importance": values})
            # Tri décroissant par importance absolue et garder top 10
            shap_df = shap_df.reindex(shap_df.Importance.abs().sort_values(ascending=False).index).head(10)
            shap_df = shap_df.sort_values("Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(8,5))
            ax.barh(shap_df["Feature"], shap_df["Importance"], color='royalblue')
            ax.set_title("Importance des variables (SHAP global)")
            ax.set_xlabel("Importance moyenne absolue")
            st.pyplot(fig)
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération des SHAP global : {e}")

elif section == "SHAP Local":
    if st.button("Afficher l'explication locale (SHAP)"):
        try:
            response = requests.get(f"{API_URL}/shap/local/{client_id}")
            response.raise_for_status()
            shap_data = response.json()

            shap_values = np.array(shap_data["shap_values"])
            expected_value = shap_data["expected_value"]
            features = shap_data["features"]  # dict {feature_name: value}

            # Préparation des données : on veut un DataFrame 1 ligne avec colonnes dans le bon ordre
            feature_names = list(features.keys())
            feature_values = [features[f] for f in feature_names]
            shap_df = pd.DataFrame([feature_values], columns=feature_names)

            explainer = shap.Explanation(
                values=shap_values,
                base_values=expected_value,
                data=shap_df.values,
                feature_names=feature_names
            )

            fig = plt.figure(figsize=(10, 5))
            shap.plots.waterfall(explainer[0], max_display=10, show=False)
            st.pyplot(fig)
            plt.close(fig)

        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la récupération des SHAP local : {e}")
        except Exception as e:
            st.error(f"Erreur lors de l'affichage SHAP local : {e}")