from pyexpat import model
import matplotlib.pyplot as plt
import pandas as pd
from ml.data import *
from ml.model import *
from settings import *
import joblib
import os
from datetime import datetime
from sklearn.model_selection import train_test_split


now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")


data = pd.read_csv(data_path)
train, test = train_test_split(data, test_size=0.20, random_state=random_state)
model = joblib.load('trained_model.joblib')
encoder = joblib.load('encoder.joblib') 
lb = joblib.load('lb.joblib') 

def fit_slice(test, category, feature):
    test_slice = test[test[category]==feature]
    X_test, y_test, _, _ = process_data(test_slice, categorical_features=cat_features, 
                                        label=label, encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    with open(slice_summary_file, "a") as f:
        f.write("Category: %s, Feature: %s\n"%(category, feature))
        f.write("Precision: %s, Recall: %s, FBeta: %s\n"%(precision, recall, fbeta))
        f.write("#"*40+"\n")

def fit_category(test, category):
    features = test[category].unique()
    for feature in features:
        fit_slice(test, category, feature)

def slice_summary(test):
    if(os.path.isfile(slice_summary_file)):
        os.remove(slice_summary_file)
    with open(slice_summary_file, 'w') as f:
        f.write("Model Slice Summary\n")
        f.write("Run at " + dt_string)
        f.write("\n\n")
    for category in cat_features:
        fit_category(test, category)

if __name__ == "__main__":
    slice_summary(test)