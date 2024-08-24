import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

url = f'https://api.sportradar.com/nhl/trial/v7/en/games/2024/01/06/schedule.json?api_key={API_KEY}'

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

data = response.json()

print(data)
