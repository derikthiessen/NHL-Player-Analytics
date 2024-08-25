import pandas as pd
import os
import datetime

seasons = [2020, 2021, 2022, 2023, 2024]
full_season_dfs = []

for season in seasons:
    url = f'https://www.hockey-reference.com/leagues/NHL_{season}_games.html'
    season_df = pd.read_html(url)

    print(season_df)


