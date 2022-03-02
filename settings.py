import subprocess
import os

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

model_path = "/starter/model/"

label = "salary"

slice_summary_file = "slice_output.txt"

# Warning, if you change these, you must run train_model.py, 
# prior to test_slices.py, or you will get overfit values
random_state = 42
test_size = 0.2

example_data = {
    'age':39,
    'workclass':'State-gov',
    'fnlgt':77516,
    'education':'Bachelors',
    'education-num':13,
    'marital-status':'Never-married',
    'occupation':'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital-gain':2174,
    'capital-loss':0,
    'hours-per-week':40,
    'native-country':'United-States'}