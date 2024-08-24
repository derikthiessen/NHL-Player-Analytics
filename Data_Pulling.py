import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

url = f"https://api.sportradar.com/nhl/trial/v7/en/seasons/2023/REG/leaders.json?api_key={API_KEY}"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)
