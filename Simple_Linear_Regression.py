import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

class Simple_Linear_Regression:
    offense_metrics = ['CF/60',
                       'FF/60',
                       'SF/60',
                       'xGF/60',
                       'SCF/60',
                       'HDCF/60',
                       'HDSF/60',
                       'MDCF/60',
                       'MDSF/60',
                       'LDCF/60',
                       'LDSF/60',
                      ]
    
    defense_metrics = ['CA/60',
                       'FA/60',
                       'SA/60',
                       'xGA/60',
                       'SCA/60',
                       'HDCA/60',
                       'HDSA/60',
                       'MDCA/60',
                       'MDSA/60',
                       'LDCA/60',
                       'LDSA/60'
                      ]
    
    shooting_metrics = ['SH%', 
                        'SV%',
                        'HDSH%',
                        'HDSV%',
                        'MDSH%',
                        'MDSV%',
                        'LDSH%',
                        'LDSV%'
                       ]
    
    def __init__(self, file_name: str = 'NHL_game_data.xlsx'):
        self.file_path = self.prepare_path(file_name)
        
        self.data = pd.read_excel(self.file_path)
        self.data = self.fix_erroneous_values(self.data)
        self.data = self.impute_missing_values(self.data)

        self.offense_metrics = self.get_offense_metrics(self.data)
        self.defense_metrics = self.get_defense_metrics(self.data)
        self.shooting_metrics = self.get_shooting_metrics(self.data)

    def prepare_path(self, file_name: str) -> str:
        load_dotenv()

        directory = os.getenv('OUTPUT_PATH')

        return os.path.join(directory, file_name)
    
    def fix_erroneous_values(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.replace('-', np.nan)
        
        for column in data.columns:
            try:
                data[column] = pd.to_numeric(data[column], errors = 'coerce')
            except TypeError:
                continue

        return data
    
    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            if data[column].isna().any() and column not in ['Game', 'Team', 'TOI', 'Season Type']:
                    mean_value = data[column].mean()
                    data[column] = data[column].fillna(mean_value)
            
        return data
    
    def get_offense_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[Simple_Linear_Regression.offense_metrics]

    def get_defense_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[Simple_Linear_Regression.defense_metrics]
    
    def get_shooting_metrics(self, data: pd.DataFrame) -> pd.DataFrame:
        return data[Simple_Linear_Regression.shooting_metrics]