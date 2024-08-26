import pandas as pd
import os
from dotenv import load_dotenv

def main():
    load_dotenv()
    
    seasons = [2020, 2021, 2022, 2023, 2024]
    regular_season_games_dfs = dict()
    postseason_games_dfs = dict()

    for season in seasons:
        start_season = season - 1
        end_season = season

        regular_season_df = extract_data(start_season, end_season, 'Regular')
        regular_season_df = clean_df(regular_season_df, 'Regular Season')
        regular_season_games_dfs[season] = regular_season_df

        postseason_df = extract_data(start_season, end_season, 'Postseason')
        postseason_df = clean_df(postseason_df, 'Postseason')
        postseason_games_dfs[season] = postseason_df
    
    all_regular_season_dfs = pd.concat(regular_season_games_dfs.values(), ignore_index=True)
    all_postseason_dfs = pd.concat(postseason_games_dfs.values(), ignore_index=True)

    combined_df = pd.concat([all_regular_season_dfs, all_postseason_dfs], ignore_index=True)

    output_path = os.getenv('OUTPUT_PATH')
    file_name = 'NHL_game_data.xlsx'
    path = os.path.join(output_path, file_name)

    combined_df.to_excel(path, index = False)

    print(f'Combined DataFrame exported to {path}')

def extract_data(start_season: int, end_season: int, season_type: str) -> pd.DataFrame:
    if season_type == 'Regular':
        value = 2
    elif season_type == 'Postseason':
        value = 3
    else:
        raise ValueError('No proper type given, please try again')
    
    url = f'https://www.naturalstattrick.com/games.php?fromseason={start_season}{end_season}&thruseason={start_season}{end_season}&stype={value}&sit=all&loc=B&team=All&rate=y'

    data = pd.read_html(url)

    return pd.DataFrame(data[0])

def clean_df(df: pd.DataFrame, season_type: str) -> pd.DataFrame:
    df['Season Type'] = season_type
    
    df['Home'] = df.apply(lambda row: determine_home(game_score = row['Game'], team = row['Team']), axis = 1)
    
    df['Win'] = df.apply(lambda row: determine_winner(game_score = row['Game'], home = row['Home']), axis = 1)
    
    df = df.drop([
             'Unnamed: 2',
             'Attendance'  
            ],
            axis = 1)

    return df

def determine_home(game_score: str, team: str) -> int:
    # Get home team last word to match
    game_score_words = game_score.split()
    home_team = game_score_words[-2]
    
    # Get input last word to see if it matches the home team last word
    team_words = team.split()
    match_team = team_words[-1]

    return 1 if home_team == match_team else 0

def determine_winner(game_score: str, home: int) -> int:
    home_score = int(game_score[-1])

    away_score_index = game_score.find(',') - 1
    away_score = int(game_score[away_score_index])

    if home:
        return 1 if home_score > away_score else 0
    else:
        return 0 if home_score > away_score else 1

main()