import requests
import json

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

response = requests.post(
    'https://salary-predictor-tg.herokuapp.com/inference', 
    uploading_data)

print(f'Status code: {response.status_code}')
print(f'Salary Prediction of a given data: {response.json()}')
