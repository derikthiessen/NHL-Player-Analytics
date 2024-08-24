import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()
API_KEY = os.getenv('API_KEY')

def get_game_ids(start_date: datetime = datetime(2019, 10, 2),
                 end_date: datetime = datetime(2024, 6, 24),
                 api_key: str = API_KEY
                 ) -> list[str]:
    
    game_ids = []
    num_days = (end_date - start_date).days
    current_day = start_date

    for i in range(1, num_days + 1):
        year = current_day.year
        month = f'{current_day.month: 02d}'
        day = f'{current_day.day: 02d}'

        url = f'https://api.sportradar.com/nhl/trial/v7/en/games/{year}/{month}/{day}/schedule.json?api_key={api_key}'

        

def extract_json_data(url):
    headers = {'accept': 'application/json'}
    response = requests.get(url, headers = headers)
    data = response.json()

    game_ids = [game['id'] for game in data['games']]

    
