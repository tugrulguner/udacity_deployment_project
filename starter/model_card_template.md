# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Random Forest Classifier from sklearn.ensemble.RandomForestClassifier was used
n_estimators = 20, and rest left as default. 

## Intended Use

This model used to predict salary from given data

## Training Data

Training and test data was splitted into 0.8 and 0.2 where both then processed by
ml.data.process_data module that involves categorical and binary label encodings

## Evaluation Data

Test data used for evaluation was the 0.2 of the dataset

## Metrics

fbeta, recall, and precision metrics were calculated using ml.model.compute_model_metrics
module. Aequitas verified that results are biased.

## Ethical Considerations

There is no ethical consideration

## Caveats and Recommendations

Try to minimize bias as much as possible