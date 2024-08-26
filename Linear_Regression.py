import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from dotenv import load_dotenv

class Linear_Regression:
    def __init__(self, file_name: str = 'NHL_game_data.xlsx', test_size: float = 0.2):
        self.file_path = self.prepare_path(file_name)

        self.data = pd.read_excel(self.file_path)

        self.dependent_variables = self.data['Win']

        self.independent_variable = self.prepare_independent_variables(self.data)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.dependent_variables,
                                                                                self.independent_variable,
                                                                                test_size = test_size,
                                                                                random_state = 1)

    def prepare_path(self, file_name: str) -> str:
        load_dotenv()

        directory = os.getenv('OUTPUT_PATH')

        return os.path.join(directory, file_name)
    
    def prepare_independent_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        column_names = data.columns.tolist()

        if 'Win' in column_names:
            column_names.remove('Win')
        
        return data[column_names]
    
    def build_model(self):
        pass
        
test = Linear_Regression()  