from fastapi.testclient import TestClient
import json

# Import our app from main.py.
from main import app, Data

# Instantiate the testing client with our app.
client = TestClient(app)

def test_api_get():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()['greeting'] == "Welcome to MLDevOPs!"

def test_api_example():
    r = client.post("/example/")
    assert r.status_code == 200
    assert r.json()['example_low']['salary'] == '<=50K'
    assert r.json()['example_high']['salary'] == '>50K'

def test_api_manual():
    r = client.post("/infer/", data=json.dumps(Data.Config.schema_extra['example_low']))
    assert r.status_code == 200
    assert r.json()['salary'] == '<=50K'