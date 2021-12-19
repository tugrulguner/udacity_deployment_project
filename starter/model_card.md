# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier from sklearn.ensemble.RandomForestClassifier was used with
n_estimators = 20, and rest left as default. 

## Intended Use

This model used to predict salary from given data

## Training Data

Training and test data was splitted into 0.8 and 0.2 where both then processed by
ml.data.process_data module that involves categorical and binary label encodings

## Evaluation Data

Test data used for evaluation was the 0.2 of the dataset

## Metrics

fbeta = 0.66
recall = 0.62
precision = 0.72
Corresponding report can be found in metric.csv in ml folder
Data Slicing based on 'Occupation' showed that 'Armed-Forces'
and 'Priv-house-serv' have precision, recall and fbeta scores 1.0.
Check the data and you can find details of this report here 
under the name slice_output.txt and csv

## Ethical Considerations

Based on the race and gender, salaries can be estimated
lower than expected, which may cause a lower level of bank loan,
which can be seem as false. Other than that, salaries may be asked
by people to kept private, here, this classification can reveal
their salary by taking some if their information that can be 
easily accessed, and therefore, may violate their request of
privacy in their salaries.

## Caveats and Recommendations

Try to minimize bias as much as possible