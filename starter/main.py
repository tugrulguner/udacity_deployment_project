
import os
from fastapi import FastAPI
import joblib
from pydantic import BaseModel
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import inference

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

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

    # Example values for the parameteres in the data
    class Config:
        schema_extra = {
            "example": {
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
        }

@app.on_event("startup")
async def startup_event(): 
    global model, encoder, label_encoder
    model = joblib.load('../starter/model/RF_Classifier.pkl')
    encoder = joblib.load('../starter/model/encoder.pkl')
    label_encoder = joblib.load('../starter/model/lb.pkl')


@app.get('/')
def welcome():
    return "Welcome to our first prediction model deployment app"


@app.post('/inference')
def inference_model(data: input_data):
    upload_data = pd.DataFrame([{
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
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

    x_test, _, _, _ = process_data(
        upload_data, categorical_features=cat_features,
        training=False, encoder=encoder, lb=label_encoder)

    predictions = inference(model, x_test)

    # Show the Classification results based on prediction
    output_result = '>50k$' if predictions[0] == 1 else '<50k$'
    return output_result
