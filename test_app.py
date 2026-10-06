import joblib


def test_model_loads():
    model = joblib.load("house_price_model.pkl")
    assert model is not None