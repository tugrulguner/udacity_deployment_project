import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from data import process_data
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot


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
_, _, encoder, lb = process_data(
  train, categorical_features=cat_features, label="salary", training=True
)
X_test, y_test, encoder, lb = process_data(
  test, categorical_features=cat_features, label="salary", training=False,
  encoder=encoder, lb=lb
)

model = joblib.load('../../model/RF_Regressor.pkl')

predictions = model.predict(X_test)


output_prediction = test[cat_features]

output_prediction['score'] = predictions
output_prediction['label_value'] = y_test

g = Group()

xtab, _ = g.get_crosstabs(output_prediction)

# print(xtab)

# output_prediction.to_csv('../../data/data_slicing_perf.csv')