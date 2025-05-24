from api.model_utils import load_model, load_client_data, predict_default

def test_prediction_format():
    model = load_model()
    df = load_client_data()
    client_id = df.index[0]
    result = predict_default(model, client_id, df)
    
    assert "probability_default" in result
    assert 0.0 <= result["probability_default"] <= 1.0
    assert result["prediction"] in [0, 1]