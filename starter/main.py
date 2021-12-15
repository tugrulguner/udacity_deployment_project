from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference

app = FastAPI()

class input_data(BaseModel):
  age: int
  workclass: str
  fnlgt: str
  education: str
  education_num: int
  marital_status: str
  occupation: str
  relationship: str
  race: str
  sex: str
  capital_gain: int
  capital_loss: int
  hours_per_week: int
  native_country: str

@app.get('/')
def welcome():
  return "Welcome to our first prediction model deployment app"

@app.post('/inference/')
def inference_model(data: input_data):
  upload_data = pd.DataFrame([{
    "age": data.age,
    "workclass": data.workclass,
    "fnlgt": data.fnglt,
    "education": data.education,
    "education-num": data.education_number,
    "marital-status": data.marital_status,
    "occupation": data.occupation,
    "relationship": data.relationship,
    "race": data.race,
    "sex": data.sex,
    "capital-gain": data.capital_gain,
    "capital-loss": data.capital_loss,
    "hours-per-week": data.hours_per_week,
    "native-country": data.native_country
  }])

  cat_features = [
  "workclass",
  "education",
  "marital-status",
  "occupation",
  "relationship",
  "race",
  "sex",
  "native-country",
  ]

  
