import pandas as pd

seasons = [2020, 2021, 2022, 2023, 2024]
full_season_dfs = dict()
full_season_dates = dict()

for season in seasons:
    url = f'https://www.hockey-reference.com/leagues/NHL_{season}_games.html'
    season_df = pd.read_html(url)

    full_season_dfs[season] = season_df

    dates = pd.to_datetime(season_df['Date'], format = '%Y-%m-%d').dt.date
    dates = pd.Series(dates).drop_duplicates().tolist()

    full_season_dates[season] = dates