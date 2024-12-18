# %%
from darts.models import Theta, ARIMA, ExponentialSmoothing
from darts import TimeSeries
from darts.utils.utils import SeasonalityMode, ModelMode


from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import set_config
set_config(transform_output = "default")

from xgboost import XGBRFRegressor, XGBRegressor
from lightgbm import LGBMRegressor




from multiprocessing import Pool
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objects as go
import missingno as msno
from plotly.colors import DEFAULT_PLOTLY_COLORS as colors


from utils import get_idle_proportion, add_time_tag
from constant import MAP_TIME_COL, NODE_TO_PARTITION_NAME


folder_path_slurm_data = Path('/projects/2/prjs1098/system_analytics_2024/slurm_data')
folder_path_prom_data = Path('/projects/2/prjs1098/system_analytics_2024/prom_data')
folder_path_EAR_data = Path('/projects/2/prjs1098/system_analytics_2024/ear_data')
folder_path_saving_results = Path('./results')



df_stat = pd.read_parquet(folder_path_slurm_data / 'sinfo_for_tabular_ML.parquet.gzip')



# WHICH TIME INTEVAL MAKES SENSE FOR ML?
time_col = 'time_4hour_interval'



TEST_DATA_LENGTH = pd.Timedelta('7days')
VAL_DATA_LENGTH = pd.Timedelta('7days')

test_upper_bound = df_stat[time_col].max()
test_lower_bound = test_upper_bound - TEST_DATA_LENGTH

val_upper_bound = test_lower_bound - MAP_TIME_COL[time_col]
val_lower_bound = val_upper_bound - VAL_DATA_LENGTH


val_mask = (val_lower_bound<=df_stat[time_col]) & (df_stat[time_col]<=val_upper_bound)
test_mask = (test_lower_bound<=df_stat[time_col]) & (df_stat[time_col]<=test_upper_bound)
train = df_stat[~(val_mask|test_mask)]
val = df_stat[val_mask]
test = df_stat[test_mask]



X_train = train.iloc[:, 3:]
y_train = train.iloc[:,0:3]

X_val = val.iloc[:,3:]
y_val = val.iloc[:,0:3]

X_test = test.iloc[:,3:]
y_test = test.iloc[:,0:3]


print(f"""val lower bound:{val_lower_bound}, val upper bound:{val_upper_bound}, test lower bound:{test_lower_bound}, 
      test lower bound: {test_upper_bound}""")
print(f"# of Training samples: {len(train)} | # of Validation samples: {len(val)} | # of Test samples: {len(test)}")
print(f"""Max Date in Train: {train[time_col].max()} | Min Date in Validation: {val[time_col].min()}|  Max Date in Validation: {val[time_col].max()}|
      Min Date in Test: {test[time_col].min()}| Max Date in Test: {test[time_col].max()}""")






""" preprocessing the data"""

categorical_column = ['hour_day', 'week_day','partition', 'last_state_lag_1']

transformer = make_column_transformer(
    # (RobustScaler(), prom_selected_signals),
    (OneHotEncoder(handle_unknown='infrequent_if_exist'), categorical_column),
    # (SimpleImputer(strategy='constant', fill_value=0), 'all'),
    remainder='passthrough')


# transoform the data
X_train_trans = transformer.fit_transform(X_train)
X_val_trans = transformer.transform(X_val)
X_test_trans = transformer.transform(X_test)
print(X_train_trans.shape, X_val_trans.shape, X_test_trans.shape)



X_train_trans = pd.DataFrame(X_train_trans).fillna(0, inplace=False)
X_val_trans = pd.DataFrame(X_val_trans).fillna(0, inplace=False)
X_test_trans = pd.DataFrame(X_test_trans).fillna(0, inplace=False)





def perform_grid_search(model_class_with_params, X_train, y_train, X_val, y_val, target_column):

    # parameter_space = list(ParameterGrid(grid_params))
    model_class = model_class_with_params['model_class']
    del model_class_with_params['model_class']
    
        
    model = model_class(**model_class_with_params)
   
    model.fit(X_train, y_train[target_column])
    
    y_val_pred = model.predict(X_val)
    score = (mean_squared_error(y_val[target_column], y_val_pred))
    print(model_class)
    # grid_search_trials = pd.DataFrame({"params": parameter_space, "score": scores}).sort_values("score")
    # best_params = grid_search_trials.iloc[0, 0]
    # best_score = grid_search_trials.iloc[0, 1]

    return {'model_class': f"{str(model).split('(')[0]}", "parameter": model_class_with_params, "score": score}


# """   
# Grid search for some hand picked ML classic models. 
# It takes time, go to the next cell, we load the results!!!
# """

param_grids = [
    {
        'model_class': [Lasso],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 10000]
    },
    {
        'model_class': [Ridge],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 10000]
    },
    
    {
        'model_class': [XGBRFRegressor],
        'n_estimators': [10,20,30],
        'colsample_bynode': [0.1, 0.3]
    },
    {
        'model_class': [XGBRegressor],
        'n_estimators': [10,20,30],
        'learning_rate': [0.01, 0.05]
    },
    {
        'model_class': [LGBMRegressor],
        'n_estimators': [10,20,30],
        'learning_rate': [0.01, 0.05, 0.1]
    }
]



# Prepare the arguments for each model
model_args = [
    (model_class_with_params, X_train_trans, y_train, X_val_trans, y_val, 'target')
    for model_class_with_params in list(ParameterGrid(param_grids))
]

# # Use a multiprocessing Pool with 3 workers
if __name__ == "__main__":
    with Pool(20) as pool:
        all_models_score_with_paramerers = pool.starmap(perform_grid_search, model_args)

print("END of the multiprocessing")
# all_models_score_with_paramerers = [perform_grid_search(*args) for args in model_args]


with open('all_models_best_params_script.pickle', 'wb') as handle:
    pickle.dump(all_models_score_with_paramerers, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    