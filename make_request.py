import requests
import json

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

r = requests.post("https://nd0821-c3-jeremy.herokuapp.com/infer/", data=json.dumps(manual_data))
print(r.status_code)
print(r.content)