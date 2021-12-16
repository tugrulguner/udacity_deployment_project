import pandas as pd
from pandas.io import pickle
import joblib
import numpy as np
from model import compute_model_metrics
from data import process_data

data = pd.read_csv('../../data/cleaned_data.csv')
model = joblib.load('../../model/RF_Classifier.pkl')
encoder = joblib.load('../../model/encoder.pkl')
label_encoder = joblib.load('../../model/lb.pkl')

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

u_val_l = []
precision_l = []
fbeta_l = []
recall_l = []

for u_val in data['occupation'].unique():
  sliced_data = data[data['occupation'] == u_val]
  
  x_test, y_test, _, _ = process_data(
    sliced_data, categorical_features=cat_features, label='salary',
    training=False, encoder=encoder, lb=label_encoder)
  
  predictions = model.predict(x_test)
  precision, recall, fbeta = compute_model_metrics(y_test, predictions)

  precision_l.append(precision)
  u_val_l.append(u_val)
  recall_l.append(recall)
  fbeta_l.append(fbeta)

u_val_l = pd.DataFrame(u_val_l, columns=['Unique Value'])
precision_l = pd.DataFrame(precision_l, columns=['Precision'])
recall_l = pd.DataFrame(recall_l, columns=['Recall'])
fbeta_l = pd.DataFrame(fbeta_l, columns=['fbeta'])
metrics = pd.concat([u_val_l, precision_l, recall_l, fbeta_l], axis=1)
metrics.to_csv('slice_output.txt', sep=' ', index=False)




