from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

manual_data = {
    'age':39,
    'workclass':'State-gov',
    'fnlgt':77516,
    'education':'Bachelors',
    'education_num':13,
    'marital_status':'Never-married',
    'occupation':'Adm-clerical',
    'relationship':'Not-in-family',
    'race':'White',
    'sex':'Male',
    'capital_gain':2174,
    'capital_loss':0,
    'hours_per_week':40,
    'native_country':'United-States'}

def test_api_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()['greeting'] == "Welcome to MLDevOPs!"

def test_api_example():
    r = client.post("/example/")
    assert r.status_code == 200
    assert r.json()['salary'] == '<=50K'

def test_api_manual():
    r = client.post("/infer/", data=json.dumps(manual_data))
    assert r.status_code == 200
    assert r.json()['salary'] == '<=50K'