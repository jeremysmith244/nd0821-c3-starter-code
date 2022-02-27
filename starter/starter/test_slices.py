from pyexpat import model
import matplotlib.pyplot as plt
import pandas as pd
from ml.data import *
from ml.model import *
from settings import *
import joblib


data = pd.read_csv(data_path)
model = joblib.load('trained_model.joblib')
encoder = joblib.load('filename.joblib') 

def get_slice(data, category, feature):
    data_slice = data[data[category]==feature]
    X_test, y_test, _, _ = process_data(data_slice, categorical_features=cat_features, 
                                        label=label, encoder=encoder, lb=lb, training=False)