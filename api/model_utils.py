import mlflow
import mlflow.pyfunc
import mlflow.lightgbm
import os
import pandas as pd

# --- Chargement modèle pyfunc (pipeline sklearn) ---
def load_model():
    model_path = "models:/Light_GBM_Best_Model/1"
    model = mlflow.pyfunc.load_model(model_path)
    return model

# --- Chargement modèle LightGBM natif (pour SHAP) ---
def load_model_lightgbm():
    model_path = "models:/Light_GBM_Best_Model/1"
    model_pyfunc = mlflow.pyfunc.load_model(model_path)
    # Accès au pipeline sklearn encapsulé dans pyfunc
    pipeline = model_pyfunc._model_impl.sklearn_model
    # Extraction du modèle LightGBM natif (dernier step nommé "model")
    model_native = pipeline.named_steps["model"]
    return model_native

# --- Chargement données clients ---
def load_client_data():
    path = os.path.join(os.path.dirname(__file__), "..", "data", "clients_test.csv")
    df = pd.read_csv(os.path.abspath(path))
    df.set_index("SK_ID_CURR", inplace=True)
    return df

# --- Conversion types numériques selon schéma modèle ---
def convert_numeric_columns_to_model_dtype(model, df):
    input_schema = model.metadata.get_input_schema()
    if input_schema is None:
        print("Attention : le modèle ne contient pas de schéma d'entrée.")
        return df
    
    type_map = {}
    for input_col in input_schema.inputs:
        col_name = input_col.name
        col_type = str(input_col.type).lower()
        if "double" in col_type or "float" in col_type:
            type_map[col_name] = 'float32'
        elif "int64" in col_type or "int" in col_type:
            type_map[col_name] = 'int32'
        else:
            type_map[col_name] = None
    
    for col, dtype in type_map.items():
        if dtype is not None and col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                print(f"Erreur conversion colonne {col} en {dtype}: {e}")
    return df

# --- Prédiction avec modèle pyfunc ---
def predict_default(model, client_id, df_clients, seuil_metier=0.5454545454545455):
    if client_id not in df_clients.index:
        return {"error": f"Client {client_id} non trouvé."}
    
    client_data = df_clients.loc[[client_id]].copy()
    client_data["SK_ID_CURR"] = client_id  # Parfois requis selon le modèle

    client_data = convert_numeric_columns_to_model_dtype(model, client_data)

    probas = model.predict(client_data)

    if hasattr(model, "best_threshold") and seuil_metier is None:
        seuil_metier = model.best_threshold
    if seuil_metier is None:
        raise ValueError("Aucun seuil métier défini.")

    prediction = int(probas[0] >= seuil_metier)

    sexe = "F" if client_data.get("CODE_GENDER_F", False).values[0] else \
           ("M" if client_data.get("CODE_GENDER_M", False).values[0] else "N/A")
    
    return {
        "SK_ID_CURR": int(client_id),
        "CODE_GENDER (Sexe)": sexe,
        "CNT_CHILDREN (Nombre d'enfants)": int(client_data["CNT_CHILDREN"].values[0]),
        "AMT_INCOME_TOTAL (revenu total)": float(client_data["AMT_INCOME_TOTAL"].values[0]),
        "probability_default": float(probas[0]),
        "prediction": prediction,
        "seuil_metier": seuil_metier
    }


# --- SHAP global ---
def get_shap_global(model_native, X_background):
    import shap
    explainer = shap.TreeExplainer(model_native)
    shap_values = explainer.shap_values(X_background)
    
    mean_shap = abs(shap_values).mean(axis=0)
    features = X_background.columns.tolist()
    
    return {
        "features": features,
        "values": mean_shap.tolist()
    }

# --- SHAP local ---
def get_shap_local(model_native, client_data):
    import shap
    explainer = shap.TreeExplainer(model_native)
    shap_values = explainer.shap_values(client_data)
    expected_value = explainer.expected_value

    return {
        "shap_values": shap_values[0].tolist() if isinstance(shap_values, list) else shap_values.tolist(),
        "expected_value": expected_value[0] if isinstance(expected_value, (list, tuple)) else expected_value,
        "features": client_data.iloc[0].to_dict()
    }

# --- Exemple d'utilisation ---

if __name__ == "__main__":
    model_pyfunc = load_model()
    model_native = load_model_lightgbm()
    df_clients = load_client_data()

    # Préparer un échantillon de données pour SHAP global (ex : les 100 premiers)
    X_background = df_clients.head(100).copy()
    X_background = convert_numeric_columns_to_model_dtype(model_pyfunc, X_background)

    # SHAP global
    shap_global = get_shap_global(model_native, X_background)
    print("SHAP global:", shap_global)

    # SHAP local sur un client
    client_id = df_clients.index[0]
    client_data = df_clients.loc[[client_id]].copy()
    client_data = convert_numeric_columns_to_model_dtype(model_pyfunc, client_data)
    shap_local = get_shap_local(model_native, client_data)
    print(f"SHAP local client {client_id}:", shap_local)

