# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from data import process_data
from model import inference
from joblib import load
from settings import *

import os

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# Instantiate the app.
app = FastAPI()

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to MLDevOPs!"}

@app.post("/infer/")
async def exercise_function(params: Data):
    
    data = {'age':params.age,
            'workclass':params.workclass,
            'fnlgt':params.fnlgt,
            'education':params.education,
            'education-num':params.education_num,
            'marital-status':params.marital_status,
            'occupation':params.occupation,
            'relationship':params.relationship,
            'race':params.race,
            'sex':params.sex,
            'capital-gain':params.capital_gain,
            'capital-loss':params.capital_loss,
            'hours-per-week':params.hours_per_week,
            'native-country':params.native_country}

    df = pd.DataFrame(data, index=[1])
    model = load(os.path.join(model_path, 'trained_model.joblib'))
    encoder = load(os.path.join(model_path, 'encoder.joblib')) 
    lb = load(os.path.join(model_path, 'lb.joblib')) 
    X_test, _, _, _ = process_data(df, categorical_features=cat_features, label=None, 
                                   encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X_test)
    salary = {'salary': lb.inverse_transform(y_pred)[0]}

    return salary

@app.post("/example/")
async def exercise_function():

    df = pd.DataFrame(example_data, index=[1])
    model = load(os.path.join(model_path, 'trained_model.joblib'))
    encoder = load(os.path.join(model_path, 'encoder.joblib')) 
    lb = load(os.path.join(model_path, 'lb.joblib'))  
    X_test, _, _, _ = process_data(df, categorical_features=cat_features, label=None, 
                                   encoder=encoder, lb=lb, training=False)
    y_pred = inference(model, X_test)
    salary = {'salary': lb.inverse_transform(y_pred)[0]}

    return salary
