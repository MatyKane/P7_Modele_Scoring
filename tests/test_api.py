from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)

def test_predict_endpoint():
    client_id = 307512  # ID présent dans df_clients
    response = client.get(f"/predict/{client_id}")
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    