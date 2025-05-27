from fastapi import FastAPI, HTTPException
from api.model_utils import (
    load_model,
    load_model_lightgbm,
    load_client_data,
    convert_numeric_columns_to_model_dtype,
    predict_default,
    get_shap_global,
    get_shap_local,
)
import sys
import os
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="API Scoring Crédit")

# Chargement modèle et données au démarrage (depuis serveur distant)
try:
    model_pyfunc = load_model()
    model_native = load_model_lightgbm()
    df_clients = load_client_data()

    # Préparation des données de fond pour SHAP global (ex : 100 premiers clients)
    X_background = df_clients.head(100).copy()
    X_background = convert_numeric_columns_to_model_dtype(model_pyfunc, X_background)

except Exception as e:
    raise RuntimeError(f"Erreur au démarrage de l'API : {e}")

@app.get("/")
def root():
    return {"message": "API prédiction risque de défaut prête"}

@app.get("/predict/{client_id}")
def predict(client_id: int, seuil: float = 0.5454545454545455):
    result = predict_default(model_pyfunc, client_id, df_clients, seuil_metier=seuil)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

@app.get("/shap/global")
def shap_global():
    try:
        X_bg = X_background.reset_index()
        shap_result = get_shap_global(model_native, X_bg)
        return shap_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP global : {e}")

@app.get("/shap/local/{client_id}")
def shap_local(client_id: int):
    if client_id not in df_clients.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")
    client_data = df_clients.loc[[client_id]].copy().reset_index()
    client_data = convert_numeric_columns_to_model_dtype(model_pyfunc, client_data)
    try:
        shap_result = get_shap_local(model_native, client_data)
        return shap_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur SHAP local : {e}")
    
# Pour exécuter localement :
# uvicorn app:app --reload --port 8000