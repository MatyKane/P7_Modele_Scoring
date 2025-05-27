import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:8000")

model = mlflow.pyfunc.load_model("models:/Light_GBM_Best_Model/latest")

print("Modèle chargé avec succès.")