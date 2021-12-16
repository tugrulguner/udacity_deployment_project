import pytest
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference


@pytest.fixture(scope='session')
def input_data():

    data = pd.read_csv('../../data/cleaned_data.csv')

    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )
    X_test, y_test, encoder, lb = process_data(
        test, categorical_features=cat_features, label="salary", training=False,
        encoder=encoder, lb=lb
    )
    return [X_train, y_train, X_test, y_test]


def test_train_model(input_data):

    model = train_model(input_data[0], input_data[1])

    prediction = model.predict(input_data[2])

    assert len(prediction) != 0


def test_compute_model_metrics(input_data):

    model = train_model(input_data[0], input_data[1])

    predictions = model.predict(input_data[2])

    precision, recall, fbeta = compute_model_metrics(
        input_data[3], predictions)

    assert precision != 0
    assert recall != 0
    assert fbeta != 0


def test_inference(input_data):

    model = joblib.load('../../model/RF_Classifier.pkl')

    inference_test = inference(model, input_data[2])

    assert len(inference_test) != 0
