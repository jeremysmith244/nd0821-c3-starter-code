# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model chosen is a simple sklearn random forest classifier. Some minor meta optimization showed n_estimators needs to be at least 200 for reasonable performance.

## Intended Use

Usage is for predicting salary classification, based on input census data containing following categories:

age	workclass	fnlgt	education	education-num	marital-status	occupation	relationship	race	sex	capital-gain	capital-loss	hours-per-week	native-country

## Training Data

Training data is taken from ./data/clean_census.csv, using sklearn.model_selection.train_test_split.
In order to maintain consistent splits, random_state and test_size are set in settings.py.

## Evaluation Data

Test data is taken from ./data/clean_census.csv, using sklearn.model_selection.train_test_split.
In order to maintain consistent splits, random_state and test_size are set in settings.py.


## Metrics

Trained model performs as follows on test data:

Precision: 0.7528691660290742, Recall: 0.6348387096774194, FBeta: 0.6888344417220862

## Ethical Considerations

Several individual categories unperform in slice analysis, so use with caution:

Category: race, Feature: Amer-Indian-Eskimo
Precision: 0.5555555555555556, Recall: 0.5, FBeta: 0.5263157894736842

Category: occupation, Feature: Farming-fishing
Precision: 0.5714285714285714, Recall: 0.14285714285714285, FBeta: 0.2285714285714286

## Caveats and Recommendations

This is not designed to be a robust model, since it is dev ops class, not a modeling class.
