import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from dotenv import load_dotenv
from statsmodels.stats.outliers_influence import variance_inflation_factor

class Logistic_Regression_Model:
    def __init__(self, file_name: str = 'NHL_game_data.xlsx', test_size: float = 0.2):
        self.file_path = self.prepare_path(file_name)

        self.data = pd.read_excel(self.file_path)
        self.data = self.fix_erroneous_values(self.data)
        self.data = self.impute_missing_values(self.data)

        # Column of the y variables
        self.dependent_variable = self.data['Win']

        # DataFrame of columns for the various x variables
        self.independent_variables = self.prepare_independent_variables(self.data)

        self.vif_data = self.calculate_vif(self.independent_variables)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.independent_variables,
                                                                                self.dependent_variable,
                                                                                test_size = test_size,
                                                                                random_state = 1)
        
        self.model = self.build_model()

        self.test_predictions = self.model.predict(self.x_test)

        self.mse = mean_squared_error(self.y_test, self.test_predictions)
        self.r2 = r2_score(self.y_test, self.test_predictions)

        self.coefficients = self.prepare_coefficients(self.model)

        self.accuracy = accuracy_score(self.y_test, self.test_predictions)

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

    def prepare_independent_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        column_names = data.columns.tolist()

        column_names.remove('Win')
        column_names.remove('Game')
        column_names.remove('Team')
        column_names.remove('TOI')
        column_names.remove('Season Type')

        column_names = self.remove_unnecessary_independent_variables(column_names)
        
        return data[column_names]
    
    def remove_unnecessary_independent_variables(self, column_names: list[str]) -> list[str]:
        for column in column_names:
            if '%' in column:
                column_names.remove(column)

        # Test removing this variable as it seems to have an outsized effect on model performance
        column_names.remove('PDO')

        # Remove these variables as they don't predict goals, they are literally goals
        column_names.remove('GF/60')
        column_names.remove('GA/60')
        column_names.remove('HDGF/60')
        column_names.remove('HDGA/60')
        column_names.remove('MDGF/60')
        column_names.remove('MDGA/60')
        column_names.remove('LDGF/60')
        column_names.remove('LDGA/60')

        return column_names

    def calculate_vif(self, independent_variables: pd.DataFrame) -> pd.DataFrame:
        vif_data = pd.DataFrame()
        vif_data["feature"] = independent_variables.columns
        vif_data["VIF"] = [variance_inflation_factor(independent_variables.values, i) for i in range(len(independent_variables.columns))]
        return vif_data

    def build_model(self) -> LogisticRegression:
        model = LogisticRegression(max_iter = 1000)

        model.fit(self.x_train, self.y_train)

        return model
    
    def prepare_coefficients(self, model: LogisticRegression) -> pd.Series:
        coefficients = pd.Series(model.coef_.flatten(), index = self.independent_variables.columns)

        sorted_coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending = False).index)

        return sorted_coefficients
        
test = Logistic_Regression_Model()
print(f'Model mse is {test.mse}', '\n\n')
print(f'Model r2 score is {test.r2}', '\n\n')
print(f'Model accuracy score is {test.accuracy}', '\n\n')
print('Model regression coefficients are:', test.coefficients, '\n\n')
print('Model variance inflation factors are:', test.vif_data)