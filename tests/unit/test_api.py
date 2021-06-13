import requests

def test_index():
    response = requests.get("http://localhost:5000/")
    assert response.status_code == 200

def test_predict():
    body = {"urls": ["https://www.thesprucepets.com/thmb/wpN_ZunUaRQAc_WRdAQRxeTbyoc=/4231x2820/filters:fill(auto,1)/adorable-white-pomeranian-puppy-spitz-921029690-5c8be25d46e0fb000172effe.jpg"]}
    response = requests.post("http://localhost:5000/predict", json=body)
    assert response.status_code == 200
    assert "data" in response.json()