
import requests

API_KEY = 'pIp1mK0nyI7T7mL7jRbvp3FdUR7zD7ypaXN4ktcl'
url = "https://api.sportradar.com/nhl/trial/v7/en/seasons/2023/REG/leaders.json?api_key=pIp1mK0nyI7T7mL7jRbvp3FdUR7zD7ypaXN4ktcl"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)
