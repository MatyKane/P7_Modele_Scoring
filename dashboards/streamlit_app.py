import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import shap
from PIL import Image
import os
import numpy as np


if st.button("Prédire le risque"):
    response = requests.get(f"{API_URL}/predict/{client_id}")
    st.write("Status code:", response.status_code)
    st.write("Contenu brut de la réponse:", repr(response.text))  # repr() pour voir caractères invisibles
    
    if response.status_code == 200:
        try:
            data = response.json()
            if "error" in data:
                st.error(data["error"])
            else:
                st.write("Résultat prédiction :")
                st.json(data)
        except Exception as e:
            st.error(f"Erreur décodage JSON : {e}")
    else:
        st.error(f"Erreur API : {response.status_code}")


        st.write("API_URL utilisée :", API_URL)