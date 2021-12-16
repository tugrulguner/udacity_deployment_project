from fastapi.testclient import TestClient
from starter.main import app
import json

with TestClient(app) as client:

    global uploading_data

    uploading_data = {
        "age": 35,
        "workclass": "Private",
        "fnlgt": 45780,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "Black",
        "sex": "Female",
        "capital_gain": 3000,
        "capital_loss": 500,
        "hours_per_week": 40,
        "native_country": "Jamaica"
    }

    uploading_data = json.dumps(uploading_data).encode('utf8')

    def test_root():
        read_get = client.get('/')
        assert read_get.status_code == 200
        assert read_get.json() == {
            'message': "Welcome to our first prediction model deployment app"
            }


    def test_inference_1():

        data_response = client.post(
            '/inference',
            data=uploading_data)
        assert data_response.json() != '<50k$' or data_response.json() != '>50k$'


    def test_inference_2():

        data_response = client.post(
            '/inference',
            data=uploading_data)
        assert data_response.json()
