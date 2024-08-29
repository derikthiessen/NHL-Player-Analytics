import pandas as pd
from Simple_Linear_Regression import (Simple_Linear_Regression as sml,
                                      offense_metrics_regression as om_reg,
                                      defense_metrics_regression as dm_reg,
                                      shooting_metrics_regression as sh_reg,
                                      save_metrics_regression as sv_reg)
from dotenv import load_dotenv
import os

def convert_to_df(regression_metrics: dict[str, sml], category: str) -> pd.DataFrame:
    keys = [key for key in regression_metrics.keys()]
    coefficients = [float(value.coefficient) for value in regression_metrics.values()]
    df_category = [category for i in range(len(regression_metrics))]
    
    return pd.DataFrame({'Metric': keys, 'Coefficient': coefficients, 'Type': df_category})

om_df = convert_to_df(om_reg, 'Offense')
dm_df = convert_to_df(dm_reg, 'Defense')
sh_df = convert_to_df(sh_reg, 'Shooting')
sv_df = convert_to_df(sv_reg, 'Save')

combined_df = pd.concat([om_df, dm_df, sh_df, sv_df], ignore_index = True)



file_name = 'Regression_Coefficients.xlsx'
load_dotenv()
directory = os.getenv('OUTPUT_PATH')

path = os.path.join(directory, file_name)

combined_df.to_excel(path, index = False)
print('Successfully saved regression coefficients')