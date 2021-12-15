# Script to train machine learning model.

import joblib
import os
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model
import pandas as pd


# Add the necessary imports for the starter code.

data = pd.read_csv('../data/cleaned_data.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label='salary',
    training=False, encoder=encoder, lb=lb)

model = train_model(X_train, y_train)

joblib.dump(model, '../model/RF_Regressor.pkl')
