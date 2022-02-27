import logging
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
from ml.data import *
from ml.model import *
from settings import *

logging.basicConfig(
    filename='model_test.log',
    level=logging.INFO,
    filemode='w')
LOGGER = logging.getLogger()

LOGGER.info("Importing data and creating model")
data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.20)

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label=label, training=True
)
X_test, y_test, _, _ = process_data(test, categorical_features=cat_features, 
                                    label=label, encoder=encoder, lb=lb, training=False)

model = train_model(X_train, y_train)
y_pred = inference(model, X_test)

LOGGER.info("Setup and model creation successful...")

def test_train_model():
    LOGGER.info("Validating model training...")
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier

def test_inference():
    LOGGER.info("Validating inference...")
    assert len(y_pred) == len(X_test)

def test_compute_metrics():
    LOGGER.info("Validating metrics...")
    precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
    assert (precision > 0) and (recall > 0) and (fbeta > 0)