import pytest
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