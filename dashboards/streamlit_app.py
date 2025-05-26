import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image
import os
import numpy as np


# Définir l'URL de l'API
API_URL = st.secrets.get("API_URL") or "http://localhost:8000"

st.title("Prédiction du risque client")

# Sélecteur ou champ de texte pour entrer un ID client
client_id = st.selectbox("Choisir un client", ["307520"])

if st.button("Prédire le risque"):
    if not client_id:
        st.warning("Veuillez entrer un ID client.")
    else:
        try:
            response = requests.get(f"{API_URL}/predict/{client_id}")
            st.write("Status code:", response.status_code)
            st.write("Réponse brute:", repr(response.text))

            if response.status_code == 200:
                try:
                    data = response.json()
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        st.write("Résultat prédiction :")
                        st.json(data)
                except Exception as e:
                    st.error(f"Erreur lors du décodage JSON : {e}")
            else:
                st.error(f"Erreur API : {response.status_code}")
        except Exception as e:
            st.error(f"Erreur lors de la requête : {e}")