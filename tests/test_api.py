from fastapi.testclient import TestClient

from dogapp.app import app
client = TestClient(app)

def test_index():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    body = {"urls": ["https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg"]}
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "data" in response.json()