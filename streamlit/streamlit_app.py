import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
import numpy as np
from PIL import Image
import os

# Configuration Streamlit
st.set_page_config(page_title="Interface Scoring Crédit", layout="wide")
st.title("Interface Scoring Crédit")

# Afficher le logo
logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
if os.path.exists(logo_path):
    image = Image.open(logo_path)
    st.image(image, width=250)
else:
    st.warning("Logo non trouvé.")

# Texte d'introduction
st.markdown("""
Bienvenue sur l'application de scoring de risque de défaut.  
            
Cette application prédit la probabilité qu'un client ne rembourse pas son crédit,  
grâce à un modèle de machine learning entraîné sur des données réelles.

Saisissez un identifiant client pour obtenir la prédiction et des explications SHAP.
""")

# Détection de l’URL de l’API (mettre avant l’appel à check_api_available)
import tomllib

API_URL = os.getenv("API_URL", "http://localhost:80")


# Vérification que l’API est bien en ligne
@st.cache_data
def check_api_available():
    try:
        resp = requests.get(f"{API_URL}/")
        return resp.status_code == 200
    except Exception:
        return False

if not check_api_available():
    st.error("API FastAPI non disponible. Veuillez réessayer plus tard.")
    st.stop()

# Navigation latérale
st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisissez une section :", ["Accueil", "Prédiction", "SHAP Global", "SHAP Local"])
st.sidebar.markdown(f" API utilisée : `{API_URL}`")

# Récupération des ID clients
@st.cache_data
def get_client_ids():
    try:
        response = requests.get(f"{API_URL}/clients")
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"Erreur lors de la récupération des ID clients : {e}")
        return []

client_ids = get_client_ids()
client_id = st.selectbox("Choisir un ID client", client_ids) if client_ids else None

# ----- Accueil -----
if section == "Accueil":
    st.write("Utilisez la barre latérale pour naviguer.")

# ----- Prédiction -----
elif section == "Prédiction":
    if st.button("Prédire le risque"):
        if client_id is None:
            st.error("Aucun ID client sélectionné.")
        elif client_id <= 0:
            st.error("Veuillez saisir un ID client valide (>0).")
        else:
            try:
                resp = requests.get(f"{API_URL}/predict/{client_id}")
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    st.error(data["error"])
                else:
                    st.subheader("Résultat de la prédiction :")
                    st.json(data)
            except Exception as e:
                st.error(f"Erreur lors de la requête API : {e}")

# ----- SHAP Global -----
elif section == "SHAP Global":
    if st.button("Afficher l'explication globale (SHAP)"):
        try:
            resp = requests.get(f"{API_URL}/shap/global")
            resp.raise_for_status()
            shap_data = resp.json()

            df_shap = pd.DataFrame({
                "Feature": shap_data["features"],
                "Importance": shap_data["values"]
            }).sort_values("Importance", ascending=True)

            top_features = df_shap.tail(10)

            fig, ax = plt.subplots()
            ax.barh(top_features["Feature"], top_features["Importance"])
            ax.set_title("Importance des variables (SHAP global)")
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Erreur lors de la récupération SHAP global : {e}")

# ----- SHAP Local -----
elif section == "SHAP Local":
    if st.button("Afficher l'explication locale (SHAP)"):
        if client_id is None:
            st.error("Veuillez sélectionner un ID client valide.")
        else:
            try:
                resp = requests.get(f"{API_URL}/shap/local/{client_id}")
                resp.raise_for_status()
                shap_data = resp.json()

                shap_values = np.array(shap_data["shap_values"])
                expected_value = shap_data["expected_value"]
                features = shap_data["features"]

                explainer = shap.Explanation(
                    values=shap_values,
                    base_values=expected_value,
                    data=pd.DataFrame([features]),
                    feature_names=list(features.keys())
                )

                shap.plots.waterfall(explainer[0], max_display=10, show=False)
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                st.error(f"Erreur lors de la récupération SHAP local : {e}")