import requests
from dotenv import load_dotenv
import os

load_dotenv() 

url = os.getenv("SOURCE_URL")
response = requests.get(url)

with open('logistic_regression_dataset.csv', 'wb') as f:
    f.write(response.content)
