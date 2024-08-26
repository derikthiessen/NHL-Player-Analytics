import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

class Linear_Regression_Model:
    def __init__(self, file_name: str = 'NHL_game_data.xlsx', test_size: float = 0.2):
        self.file_path = self.prepare_path(file_name)

        self.data = pd.read_excel(self.file_path)
        self.data = self.replace_missing_values(self.data)
        self.data = self.impute_missing_values(self.data)

        # Column of the y variables
        self.dependent_variable = self.data['Win']

        # DataFrame of columns for the various x variables
        self.independent_variables = self.prepare_independent_variables(self.data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent_variables,
                                                                                self.dependent_variable,
                                                                                test_size = test_size,
                                                                                random_state = 1)
        
        self.model = self.build_model()

        self.test_predictions = self.model.predict(self.x_test)

        self.mse = mean_squared_error(self.y_test, self.test_predictions)
        self.r2 = r2_score(self.y_test, self.test_predictions)

        print(f'The mean squared error of the model is: {self.mse}')
        print('\n\n')
        print(f'The r2 score of the model is: {self.r2}')

    def prepare_path(self, file_name: str) -> str:
        load_dotenv()

        directory = os.getenv('OUTPUT_PATH')

        return os.path.join(directory, file_name)
    
    def replace_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.replace('-', np.nan)

    def impute_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            if data[column].isna().any() and column not in ['Game', 'Team', 'TOI', 'Season Type']:
                try:
                    mean_value = data[column].mean()
                    data[column].fillna(mean_value, inplace = True)
                except TypeError:
                    print(f'Could not calculate mean on {column}')
                    break
            
        return data
    
    '''
    To do: some columns in excel are being stored as text when they should be numeric, 
    need to ensure that numeric columns are being stored as such
    '''

    def prepare_independent_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        column_names = data.columns.tolist()

        column_names.remove('Win')
        column_names.remove('Game')
        column_names.remove('Team')
        column_names.remove('TOI')
        column_names.remove('Season Type')
        
        return data[column_names]
    
    def build_model(self) -> LinearRegression:
        model = LinearRegression()

        model.fit(self.x_train, self.y_train)

        return model
        
test = Linear_Regression_Model()  