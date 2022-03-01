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

r = requests.post("http://127.0.0.1:8000/infer/", data=json.dumps(manual_data))

# for tent in r.iter_content():
#     print(tent)

print(r.json())