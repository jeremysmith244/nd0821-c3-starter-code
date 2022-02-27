# Script to train machine learning model.

import joblib
from sklearn.model_selection import train_test_split
import pandas as pd
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference
from joblib import dump
from settings import *

# Add the necessary imports for the starter code.

if __name__ == '__main__':

    # Add code to load in the data.
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label=label, training=True
    )
    X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                        label=label, encoder=encoder, lb=lb, training=False)
    # Proces the test data with the process_data function.

    model = train_model(X_train, y_train)
    y_pred = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    # Train and save a model.

    joblib.dump(model, 'trained_model.joblib')
    joblib.dump(encoder, 'encoder.joblib')
    joblib.dump(encoder, 'lb.joblib')

