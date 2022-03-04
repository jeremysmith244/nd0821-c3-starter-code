import requests
import json

from main import Data

test_data = Data.Config.schema_extra['example_low']
print(test_data)
r = requests.post("http://127.0.0.1:8000/infer/", data=json.dumps(test_data))
print(r.json())