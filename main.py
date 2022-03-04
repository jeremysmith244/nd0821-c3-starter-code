# Put the code for your API here.#

from fastapi import FastAPI
from pydantic import BaseModel, Field
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
    education_num: int = Field(alias='education-num')
    marital_status: str = Field(alias='marital-status')
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias='capital-gain')
    capital_loss: int = Field(alias='capital-loss')
    hours_per_week: int = Field(alias='hours-per-week')
    native_country: str = Field(alias='native-country')

    class Config:
        schema_extra = {
            "example_low": {
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
                'native-country':'United-States'
            },
            "example_high": {
                'age':52,
                'workclass':'Self-emp-incv',
                'fnlgt':287927,
                'education':'HS-grad',
                'education-num':9,
                'marital-status':'Married-civ-spouse',
                'occupation':'Exec-managerial',
                'relationship':'Wife',
                'race':'White',
                'sex':'Female',
                'capital-gain':15024,
                'capital-loss':0,
                'hours-per-week':40,
                'native-country':'United-States'
            }
        }

@app.get("/")
async def say_hello():
    return {"greeting": "Welcome to MLDevOPs!"}

@app.post("/infer/")
async def exercise_function(params: Data):

    df = pd.DataFrame(params.dict(by_alias=True), index=[1])
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

    df_low = pd.DataFrame(Data.Config.schema_extra['example_low'], index=[1])
    df_high = pd.DataFrame(Data.Config.schema_extra['example_high'], index=[1])

    model = load(os.path.join(model_path, 'trained_model.joblib'))
    encoder = load(os.path.join(model_path, 'encoder.joblib')) 
    lb = load(os.path.join(model_path, 'lb.joblib'))

    X_test_low, _, _, _ = process_data(df_low, categorical_features=cat_features, label=None, 
                                   encoder=encoder, lb=lb, training=False)
    y_pred_low = inference(model, X_test_low)
    salary_low = {'salary': lb.inverse_transform(y_pred_low)[0]}
    X_test_high, _, _, _ = process_data(df_high, categorical_features=cat_features, label=None, 
                                   encoder=encoder, lb=lb, training=False)
    y_pred_high = inference(model, X_test_high)
    salary_high = {'salary' : lb.inverse_transform(y_pred_high)[0]}
    
    examples = {
        'example_low': salary_low,
        'example_high': salary_high
    }
    return examples
