import pytest
from fastapi import FastAPI
import requests

test_app = FastAPI()

@test_app.get('/')
def test_welcome():
  read_get = requests.get('localhost:/8000/')
  assert read_get.status_code == '200'