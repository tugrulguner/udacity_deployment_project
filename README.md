## This is a Salary Prediction App by Tugrul Guner

Salary data is used from the publicly available Census Bureau data. 
Raw and cleaned data can be found in starter/data folder. They are tracked by DVC.

Data cleaning is as simple as possibly be, where nan and empty spaces in the strings
were just cleared. 

data.py in the starter/starter/ml/ contains the process_data function that uses
one hot encoder and label binarizer from sklearn to encode given categorical features.
You can find encoder and label binarizer models saved in starter/model folder as .pkl files

model.py in the starter/starter/ml contains three functions that we use for training, computing
metrics and inference. Random Forest Classifier with n_estimators=20 (rest left as default) is trained
over the processed data (using train_model.py in starter/starter/) by process_data where train and 
test sets were separated with 0.2 using train_test_split by sklearn. You can find details about the 
model and metrics in model_card. Random Forest Classifier model saved as RF_Classifier.pkl 
into the starter/model folder.

Recall, precision, fbeta metrics were calculated over the test data and metrics.csv in the starter/stater folder 
and model_card contains the metric results.

To understand the bias in the model, model slicing method is applied to 'occupation' feature where metrics
were calculated over all of its unique values. It is found that there are two values that need to be
taken care of carefully in order to avoid from bias, which again details can be found model_card.
You can check the data from slice_output.csv and slice_output.txt in starter/starter/ml folder.

We are applying two pytest:
   * To check all three functions in model.py (test_model.py)
   * To check API for both GET and POST methods (test_api.py)

FastAPI is used for the deployment and together with the Continous Deployment and Continous Integration
(provided by Git Actions), this repo connected to Heroku with all of these options.

During the continous integration, GitHub checks flake8, pytest, and general CI, which is then deployed
to Heroku when any push to GitHub is made passes all of these.

You can find the app: [Salary Predictor App](https://salary-predictor-tg.herokuapp.com/)

You can send a post request to receive any prediction:

### Example data - you can change the values
```
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
```

### json encoding

uploading_data = json.dumps(uploading_data).encode('utf8')

### Sending post request to corresponding url-end with correct data

response = requests.post(
    'https://salary-predictor-tg.herokuapp.com/inference', 
    uploading_data)

### Then you can print the prediction result

print(f'Salary Prediction of a given data: {response.json()}')

### OR you can just simply use live_post.py python script in starter folder