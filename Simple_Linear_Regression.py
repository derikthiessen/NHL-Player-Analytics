import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

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
                        'HDSH%',
                        'MDSH%',
                        'LDSH%'
                       ]
    
    save_metrics = ['SV%',
                    'HDSV%',
                    'MDSV%',
                    'LDSV%'
                    ]
    
    def __init__(self, metric: str, file_name: str = 'NHL_game_data.xlsx', test_size: float = 0.2):
        self.file_path = self.prepare_path(file_name)
        
        self.data = pd.read_excel(self.file_path)
        self.data = self.fix_erroneous_values(self.data)
        self.data = self.impute_missing_values(self.data)

        self.metric = metric
        
        self.dependent_variable = self.prepare_dependent_variable(self.metric, self.data)

        self.independent_variable = self.data[[self.metric]]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent_variable,
                                                                                self.dependent_variable,
                                                                                test_size = test_size,
                                                                                random_state = 1)
        
        self.model = self.build_model()

        self.test_predictions = self.model.predict(self.x_test)

        self.mse = mean_squared_error(self.y_test, self.test_predictions)
        self.r2 = r2_score(self.y_test, self.test_predictions)
        self.coefficient = self.model.coef_


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
    
    def prepare_dependent_variable(self, metric: str, data: pd.DataFrame) -> pd.Series:
        if metric in Simple_Linear_Regression.offense_metrics:
            return data['GF']
        elif metric in Simple_Linear_Regression.defense_metrics:
            return data['GA']
        elif metric in Simple_Linear_Regression.shooting_metrics:
            return data['GF Above Expected']
        elif metric in Simple_Linear_Regression.save_metrics:
            return data['GA Above Expected']
        else:
            raise ValueError(f'Input metric {metric} not present in list of metrics')

    def build_model(self) -> LinearRegression:
        model = LinearRegression()

        model.fit(self.x_train, self.y_train)

        return model

'''
Output code
'''

# Offense metrics
offense_metrics_regression = dict()
for metric in Simple_Linear_Regression.offense_metrics:
    model = Simple_Linear_Regression(metric)
    offense_metrics_regression[metric] = model

# Defense metrics
defense_metrics_regression = dict()
for metric in Simple_Linear_Regression.defense_metrics:
    model = Simple_Linear_Regression(metric)
    defense_metrics_regression[metric] = model

# Shooting metrics
shooting_metrics_regression = dict()
for metric in Simple_Linear_Regression.shooting_metrics:
    model = Simple_Linear_Regression(metric)
    shooting_metrics_regression[metric] = model

# Save metrics
save_metrics_regression = dict()
for metric in Simple_Linear_Regression.save_metrics:
    model = Simple_Linear_Regression(metric)
    save_metrics_regression[metric] = model

print('Successfully built all models')