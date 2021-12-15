# import pytest
from fastapi import FastAPI
import requests
import json

test_app = FastAPI()

@test_app.get('/')
def test_root():
  read_get = requests.get('http://127.0.0.1:8000/')
  assert read_get.status_code == 200

@test_app.get('/')
def test_welcome():
  read_get_message = requests.get('http://127.0.0.1:8000/')
  assert len(read_get_message.json()[0]) != 0

@test_app.post('/inference_check')
def test_inference_1():
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
  data_response = requests.post('http://127.0.0.1:8000/inference', uploading_data)
  assert data_response.json() != '<50k$' or data_response.json() != '>50k$'

@test_app.post('/inference_check_2')
def test_inference_2():
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
  data_response = requests.post('http://127.0.0.1:8000/inference', uploading_data)
  assert data_response.json()