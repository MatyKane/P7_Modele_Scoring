import mlflow

def test_model_loading():
    mlflow.set_tracking_uri("http://127.0.0.1:5000") # ou serveur distant
    model = mlflow.pyfunc.load_model("models:/Light_GBM_Best_Model/latest")
    assert model is not None