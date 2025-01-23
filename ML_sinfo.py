# %%
from darts.models import Theta, ARIMA, ExponentialSmoothing
from darts import TimeSeries
from darts.utils.utils import SeasonalityMode, ModelMode


from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import ParameterGrid, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


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


from utils import get_idle_proportion, add_time_tag, get_jobs_data
from constant import MAP_TIME_COL, NODE_TO_PARTITION_NAME

# from sklearn import set_config
# set_config(transform_output = "pandas")

# %% [markdown]
# 
# # Classic ML and Econometric models training

# %%
# pd.set_option('print.max_columns', None)
# pd.set_option('print.max_rows', None)
# pd.options.print.precision = 4 # show 4 digits precision
folder_path_slurm_data = Path('/projects/2/prjs1098/system_analytics_2024/slurm_data')
folder_path_prom_data = Path('/projects/2/prjs1098/system_analytics_2024/prom_data')
folder_path_EAR_data = Path('/projects/2/prjs1098/system_analytics_2024/ear_data')
folder_path_saving_results = Path('./results')


# for getting the up-to-date data run the clenaing_sinfo file
# with the latest data from the system.
file_sinfo = 'sinfo_cleaned_2025-01-09.parquet.gzip'
df = pd.read_parquet(folder_path_slurm_data / file_sinfo)


# this takes a bit of time, so we instead load the result
# df_prom_average = get_prom_average_node_sinfo(all_prom_file_paths[0:2], map_time_col[time_col], time_col)
df_prom_average = pd.read_parquet(folder_path_prom_data /'average_signal_prom'
                                  /'prom_average_data_2025-01-09.parquet.gzip')

# get the latest tables from the EAR data base, this is not up to date. run get_ear_db_data ==> I wish I could change this!
jobs_table_path = folder_path_EAR_data / 'jobs_2025-01-07.parquet.gzip'
apps_table_path = folder_path_EAR_data / 'applications_2025-01-07.parquet.gzip'
df_job_number_history= get_jobs_data(jobs_table_path, apps_table_path)
print(df.sample(n=5), df_prom_average.sample(n=5), df_job_number_history.sample(n=5))




TRAIN_MODE = True
# if TRAIN_MODE is False, ensure that the file exist in the work space
trained_model_parameters = 'all_models_best_params_Jan_09.pickle'

# %%
# remove the latest date from sinfo to align with promehtues
df = add_time_tag(df)
# print a sample
N = len(df)//20
print(df.iloc[N: N + 5, :])

# %%
"""    
PICK THE TIME INTERVAL THAT WE WANT TO GROUP DATA.
"""

# WHICH TIME INTEVAL MAKES SENSE FOR ML?
time_col = 'time_4hour_interval'

df_stat, df_idle, df_total = get_idle_proportion(df, time_col)
# show a smaple
initial_data_size= len(df_stat)
print(df_stat.sample(n=5))
print(f"initial data size: {initial_data_size}")
print(f"Is there duplication in node and time: {df_stat[['node', time_col]].duplicated().any()}")

# note the trick here! this happens again in the future for Promethues data
df_last = df.groupby(['node', time_col], as_index=False)[['node', time_col, 'state', 'time']].tail(1).copy()
# merge it with the main
df_stat = pd.merge(df_stat, df_last[['node', time_col, 'state', 'time']],
                   how='left', on=['node', time_col]).copy()
# rename it
df_stat.rename(columns={'time':'time_for_last_state', 'state': 'last_state'}, inplace=True)
# show a sample
print(df_stat.sample(n=10))
print(f"Is there duplication in node and time: {df_stat[['node', time_col]].duplicated().any()}")

df_stat.sort_values(['node', time_col], inplace=True)
# drop some columns
df_stat.drop(['idle_duration', 'all_state_durations_in_interval'],
             axis=1, inplace=True)
print(df_stat.head())

# %%
# pivot the table for plotting, call the function for getting longer intervals
df_stat_pivot = pd.pivot_table(df_stat, index=time_col, columns=['node'], values=['idle_proportion'])
df_stat_pivot = df_stat_pivot.droplevel(level=0, axis=1).copy()
# df_stat_pivot.index = df_stat_pivot.index.strftime("%Y-%m-%d %H:%M")
df_stat_pivot.index = df_stat_pivot.index.strftime("%Y-%m-%d")

print(f"""Top 5 nodes with nan values: 
      {df_stat_pivot.isna().sum().sort_values(ascending=False).head()}""")
node_names_with_high_nan_values = df_stat_pivot.isna().sum().sort_values(ascending=False).index.to_list()
df_sorted_nodes = df_stat_pivot[node_names_with_high_nan_values].copy()
msno.matrix(df_sorted_nodes.iloc[:, 0:10])

# we have nan values in the idle proportion!==> it seems that this has happened recently and two of the nodes are not
# showing any state values

# %%
"""  
Drop the nodes that have really bad nan values
"""
mask = (df_stat['node'] == 'gcn56') | (df_stat['node'] == 'gcn25')
df_stat = df_stat[~mask].copy()


# %% [markdown]
# # Feature engineering
# 
# 
# Add new features to the data
# 

# %%
# add the lags here; 
time_sample_in_one_day = pd.Timedelta('1day') // MAP_TIME_COL[time_col]

lags = (
    (np.arange(3) + 1).tolist()
    + (np.arange(3) + (time_sample_in_one_day-2)).tolist()
    + (np.arange(3) + (time_sample_in_one_day * 7) - 2).tolist()
)

for i in lags:
    df_stat[f'lag_{i}'] = df_stat.groupby('node', as_index=False)['idle_proportion'].shift(i)
    

# adding aggreation to the signals
rolls = [3 * time_sample_in_one_day, 2 * time_sample_in_one_day,  time_sample_in_one_day]
agg_functions = ['mean', 'std']

for agg_function in agg_functions:
    for roll in rolls:
        df_stat[f'roll_{roll}_{agg_function}'] = (df_stat
                                                  .groupby('node', as_index=False)['idle_proportion']
                                                  .shift(1) # shift to avoid leakage!
                                                  .rolling(roll)
                                                  .agg(agg_function))

# add calender feature like day of the week and month
df_stat['hour_day'] = df_stat[time_col].dt.hour
df_stat['week_day'] = df_stat[time_col].dt.day_of_week
# df_stat['month'] = df_stat[time_col].dt.month


# add parition name, use the map from constant
df_stat['partition'] = (df_stat['node']
                        .apply(lambda x: NODE_TO_PARTITION_NAME
                               .get(x, 'other')))

# add the number that comes after the node as a feature
df_stat['node_number'] = df_stat['node'].str.split('n').str[1].astype(int)


# shifting the last state to avoid data leakage.
df_stat['last_state_lag_1'] = (df_stat
                               .groupby('node', as_index=False)['last_state']
                               .shift(1))
# print some rows   
N = np.random.randint(len(df_stat) - 10)
df_stat.iloc[N:N+10, :]

# %%
"""   
We bring the data about the number of jobs that happend before at the same day
last year or last quarter ...
"""

df_temp = df_job_number_history.groupby(['job_start_time_date'], as_index=False)['job_id'].aggregate(['nunique'])
df_temp['job_start_time_date'] = pd.to_datetime(df_temp['job_start_time_date'])
df_temp.rename(columns={'nunique': 'number_of_jobs'}, inplace=True)


# add this tag to do a merge
df_stat['date'] = pd.to_datetime(df_stat[time_col].dt.date)
df_stat['shift_date_one_year'] = pd.to_datetime((df_stat['date']  -  pd.DateOffset(years=1)))
df_stat['shift_date_two_year'] = pd.to_datetime(df_stat['date']  -  pd.DateOffset(years=2))
df_stat['shift_date_one_day'] = pd.to_datetime((df_stat['date']  -  pd.Timedelta('1day')))
df_stat['shift_date_one_week'] = pd.to_datetime(df_stat['date']  -  pd.Timedelta('7day'))
df_stat['shift_date_two_week'] = pd.to_datetime(df_stat['date']  -  pd.Timedelta('14day'))




map_shift_data_job= {'shift_date_one_year': 'number_of_jobs_1_year_ago',
 'shift_date_two_year': 'number_of_jobs_2_year_ago',
 'shift_date_one_day': 'number_of_jobs_1_day_ago',
 'shift_date_one_week': 'number_of_jobs_7_day_ago',
 'shift_date_two_week': 'number_of_jobs_14_day_ago',
 }




for key, val in map_shift_data_job.items():
    df_stat = df_stat.merge(df_temp, 
                        how='left', 
                        left_on=key, 
                        right_on='job_start_time_date')
    df_stat.rename(columns={'number_of_jobs': val}, inplace=True)
    df_stat.drop('job_start_time_date', axis=1, inplace=True)

print(df_stat.sample(n=5))

# %%
""" 
ENRICHING THE DATA SET:
Bring Promethues data set and get average of the measurement for the signals 
This possibly shows that when the node is not idle to which extent it was working.
"""

prom_selected_signals = [
'node_cpu_frequency_hertz_mean', 'node_filesystem_files_mean',
'node_cpu_frequency_max_hertz_mean',
 'node_cpu_frequency_min_hertz_mean',
 'node_cpu_package_throttles_total_mean',
 'node_disk_io_now_mean',
 'node_disk_read_bytes_total_mean',
 'node_disk_writes_completed_total_mean',
 'node_disk_written_bytes_total_mean',
 'node_filesystem_avail_bytes_mean',
 'node_filesystem_files_free_mean',
 'node_filesystem_free_bytes_mean',
  'node_network_receive_bytes_total_mean',
 'node_network_receive_drop_total_mean',
 'node_network_receive_multicast_total_mean',
 'node_network_receive_packets_total_mean',
 'node_network_transmit_bytes_total_mean',
 'node_network_transmit_packets_total_mean',
  'node_procs_blocked_mean',
 'node_procs_running_mean',
 'node_rapl_core_joules_total_mean',
 'node_rapl_dram_joules_total_mean',
 'node_rapl_package_joules_total_mean',
 'surf_ambient_temp_mean',
 'surf_confluent_mean',
 'surf_cpu_power_mean',
 'surf_exhaust_temp_mean',
 'surf_gpu_board_power_mean',
 'surf_inlet_water_temp_mean',
 'surf_mem_power_mean',
 'surf_sys_power_mean',
 'surf_temp_cpu_mean',
 'surf_virtual_mean',
 'up_mean']




df_prom_average[time_col] = (df_prom_average['time']).dt.floor(freq=MAP_TIME_COL[time_col])
# then do an average to get the result that you want
df_prom_average_interval = (df_prom_average.groupby(['node', time_col], as_index=False)
                                            .mean(numeric_only=True).copy())
print(df_prom_average_interval[prom_selected_signals].head())

print(f"Is there duplication in node and time: {df_prom_average_interval[['node', time_col]].duplicated().any()}")


#### do the merge finally here!
# add dummy time to prevent leakage
df_stat['shift_time_col'] = df_stat[time_col] - MAP_TIME_COL[time_col]
df_stat = pd.merge(df_stat, df_prom_average_interval[['node', time_col] + prom_selected_signals],
                   how='left', left_on=['node', 'shift_time_col'], right_on=['node', time_col],
                   suffixes=['', '_y']).copy()
print(df_stat.sample(n=5))
print(f"Is there duplication in node and time: {df_stat[['node', time_col]].duplicated().any()}")



# %%
"""   
Dropping all the columns that we think might not be useful.
"""

df_stat.drop((['date', 'shift_time_col', f"{time_col}_y"] 
              + list(map_shift_data_job.keys()) + ['last_state', 'time_for_last_state']),
             axis=1, inplace=True)


# rename idle_proportion to target
df_stat.rename(columns={'idle_proportion':'target'}, inplace=True)
# save the data here:
# df_stat.to_parquet(folder_path_slurm_data / 'sinfo_for_tabular_ML.parquet.gzip')

print(df_stat.sample(n=5))

# %% [markdown]
# # Train Test Valildation Split
# 
# 
# The final week in the data for test and the week before that is for validation
# 

# %%
TEST_DATA_LENGTH = pd.Timedelta('7days')
VAL_DATA_LENGTH = pd.Timedelta('7days')
MAX_DATE_OFFSET = pd.Timedelta('3days') # our sinfo is usually up-to-date, but data from other sources are not


test_upper_bound = (df_stat[time_col].max()) - MAX_DATE_OFFSET
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




# %% [markdown]
# # Preprocessing the data
# 
# 
# 

# %%
categorical_features = ['hour_day', 'week_day','partition', 'last_state_lag_1']
numerical_exogenous_features = list(map_shift_data_job.values()) + prom_selected_signals
numerical_endogenous_features = list(set(X_train.columns) - set(numerical_exogenous_features) 
                                     - set(categorical_features))


categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist"))
    ]
)
numeric_exogenous_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="constant", fill_value=0)), ("scaler", StandardScaler())]
)

numeric_endogenous_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median"))]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num_exo", numeric_exogenous_transformer, numerical_exogenous_features),
        ("num_endo", numeric_endogenous_transformer, numerical_endogenous_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder='passthrough'
)

# transoform the data
X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans = preprocessor.transform(X_val)
X_test_trans = preprocessor.transform(X_test)
print(X_train_trans.shape, X_val_trans.shape, X_test_trans.shape)


# %%
def perform_grid_search(model_class_with_params, X_train, y_train, X_val, y_val, target_column):

    # parameter_space = list(ParameterGrid(grid_params))
    model_class = model_class_with_params['model_class']
    del model_class_with_params['model_class']
    
        
    model = model_class(**model_class_with_params)
    print(model_class)
    model.fit(X_train, y_train[target_column])
    
    y_val_pred = model.predict(X_val)
    score = (mean_squared_error(y_val[target_column], y_val_pred))
    
    # grid_search_trials = pd.DataFrame({"params": parameter_space, "score": scores}).sort_values("score")
    # best_params = grid_search_trials.iloc[0, 0]
    # best_score = grid_search_trials.iloc[0, 1]

    return {'model_class': f"{str(model).split('(')[0]}", "parameter": model_class_with_params, "score": score}



def classic_eco_training(model, model_name, y_train, y_val, y_test):
    
    # pivot the table for plotting, call the function for getting longer intervals
    y_train_piv = pd.pivot_table(y_train[['node', time_col, 'target']],
                                 index=time_col, columns=['node'], values=['target'])
    y_train_piv = y_train_piv.droplevel(level=0, axis=1).copy()


    y_val_piv = pd.pivot_table(y_val[['node', time_col, 'target']],
                               index=time_col, columns=['node'], values=['target'])
    y_val_piv = y_val_piv.droplevel(level=0, axis=1).copy()

    y_test_piv = pd.pivot_table(y_test[['node', time_col, 'target']], 
                                index=time_col, columns=['node'], values=['target'])
    y_test_piv = y_test_piv.droplevel(level=0, axis=1).copy()

    # transform the data suitable for dart library
    train_series_for_validation = [TimeSeries.from_dataframe(y_train_piv, value_cols=col)
                                   for col in y_train_piv.columns]
    
    y_train_for_test = pd.concat([y_train_piv, y_val_piv], axis=0)
    train_series_for_test = [TimeSeries.from_dataframe(y_train_for_test, value_cols=col)
                                for col in y_train_for_test.columns]
    
    # val_series = [TimeSeries.from_dataframe(y_val_piv, value_cols=col) for col in y_val_piv.columns]
    # test_series = [TimeSeries.from_dataframe(y_test_piv, value_cols=col) for col in y_test_piv.columns]


    theta_val_pred = []
    theta_test_pred = []
    # train is df
    for series in (train_series_for_validation):
        # print(series.values())
        model.fit(series)
        pred = model.predict(len(y_val_piv))
        theta_val_pred.append(pred)
        

    for series in (train_series_for_test):
        # print(series.values())
        model.fit(series)
        pred = model.predict(len(y_test_piv))
        theta_test_pred.append(pred)
        
        
    df_theta_val_pred = pd.concat([pd.Series(p.values().flatten()) for p in theta_val_pred], axis=1) 
    df_theta_val_pred.index, df_theta_val_pred.columns = y_val_piv.index, y_val_piv.columns

    df_theta_test_pred = pd.concat([pd.Series(p.values().flatten()) for p in theta_test_pred], axis=1) 
    df_theta_test_pred.index, df_theta_test_pred.columns = y_test_piv.index, y_test_piv.columns



    df_theta_val_pred.reset_index(inplace=True)
    df_theta_val_pred = df_theta_val_pred.melt(id_vars=time_col)

    df_theta_test_pred.reset_index(inplace=True)
    df_theta_test_pred = df_theta_test_pred.melt(id_vars=time_col)

    y_val = y_val.merge(df_theta_val_pred, how='left', on=['node', time_col])
    y_test = y_test.merge(df_theta_test_pred, how='left', on=['node', time_col])
    y_val.rename(columns={'value':model_name}, inplace=True)
    y_test.rename(columns={'value':model_name}, inplace=True)
    return y_val, y_test


# %%
"""    
                            Naive one step ahead prediction, 

use the previous value as your prediction
Naive seasnoal prediction: get the value of previous week excatly the same time and say that this is my prediction
"""
y_train['label'] = 'train'
y_val['label'] = 'val'
y_test['label'] = 'test'

train_val_test_concat = pd.concat([y_train, y_val, y_test], axis=0)
train_val_test_concat.sort_values(['node', time_col], inplace=True)

train_val_test_concat['shifted_target'] = train_val_test_concat.groupby('node')['target'].shift(1)
train_val_test_concat['shifted_target_week'] = (train_val_test_concat
                                                .groupby('node')['target']
                                                .shift(time_sample_in_one_day * 7))


# the first model
y_test['Naive'] = train_val_test_concat.loc[train_val_test_concat['label'] == 'test', 'shifted_target']
y_test['Naive seasonal'] = train_val_test_concat.loc[train_val_test_concat['label'] == 'test', 'shifted_target_week']


# save this for later use and training a meta model
y_val['Naive'] = train_val_test_concat.loc[train_val_test_concat['label'] == 'val', 'shifted_target']
y_val['Naive seasonal'] = train_val_test_concat.loc[train_val_test_concat['label'] == 'val', 'shifted_target_week']

# drop the labels
y_train.drop(columns='label', inplace=True)
y_val.drop(columns='label', inplace=True)
y_test.drop(columns='label', inplace=True)


# %%
"""
                      *** Econometric  models ***
"""

# model = Theta(theta=2, season_mode=SeasonalityMode.ADDITIVE)
# y_val, y_test = classic_eco_training(model, 'Theta', y_train, y_val, y_test)

# model = ExponentialSmoothing(trend=ModelMode.ADDITIVE, seasonal=SeasonalityMode.ADDITIVE)
# y_val, y_test = classic_eco_training(model, 'Exponential', y_train, y_val, y_test)

# model = ARIMA(p=2, d=1, q =1)
# y_val, y_test = classic_eco_training(model, 'ARIMA', y_train, y_val, y_test)


# %%
# """   
# Grid search for some hand picked ML classic models. 
# It takes time, go to the next cell, we load the results!!!
# """


NUM_ESTIMATORS = list(range(100, 2100, 100)) + [10, 50]
LEARNING_RATES = np.concatenate((np.arange(0.1, 1, 0.1), np.linspace(0.001, 0.099, 5)))
COL_SAMPLES = np.concatenate((np.arange(0.1, 1, 0.1), np.linspace(0.001, 0.099, 5)))
ALPHAS = np.linspace(0.0001, 10**4, 20)


param_grids = [
    {
        'model_class': [Lasso],
        'alpha': ALPHAS
    },
    {
        'model_class': [Ridge],
        'alpha': ALPHAS
    },
    {
        'model_class': [XGBRFRegressor],
        'n_estimators': NUM_ESTIMATORS,
        'colsample_bytree': COL_SAMPLES,
        'device': ['cuda']
    },
    {
        'model_class': [XGBRegressor],
        'n_estimators': NUM_ESTIMATORS,
        'learning_rate': LEARNING_RATES,
        'device': ['cuda']
    },
    {
        'model_class': [LGBMRegressor],
        'n_estimators': NUM_ESTIMATORS,
        'learning_rate': LEARNING_RATES,
        # 'device': ['cuda']
    }
]




# Prepare the arguments for each model
model_args = [
    (model_class_with_params, X_train_trans, y_train, X_val_trans, y_val, 'target')
    for model_class_with_params in list(ParameterGrid(param_grids))
]

# # Use a multiprocessing Pool with 3 workers
# if __name__ == "__main__":
#     if TRAIN_MODE:
#             with Pool(50) as pool:
#                 all_models_score_with_paramerers = pool.starmap(perform_grid_search, model_args)
#         # all_models_score_with_paramerers = [perform_grid_search(*args) for args in model_args]
#             with open(trained_model_parameters, 'wb') as handle:
#                 pickle.dump(all_models_score_with_paramerers, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
if TRAIN_MODE:
    all_models_score_with_paramerers = [perform_grid_search(*args) for args in model_args]
    with open(trained_model_parameters, 'wb') as handle:
        pickle.dump(all_models_score_with_paramerers, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    

# %%
with open(trained_model_parameters, 'rb') as handle:
    all_models_score_with_paramerers = pickle.load(handle)
    
df_all_models_score_with_paramerers = pd.DataFrame(all_models_score_with_paramerers) 
df_model_score_best = (df_all_models_score_with_paramerers
                       .sort_values(by=['model_class', 'score'])
                       .groupby('model_class')
                       .head(1))
print(df_all_models_score_with_paramerers, df_model_score_best)
map_name_to_class = {'LGBMRegressor':LGBMRegressor,
                     'Lasso':Lasso,
                     'Ridge':Ridge,
                     'XGBRFRegressor':XGBRFRegressor,
                     'XGBRegressor':XGBRegressor
                     }

for i, model_spec in df_model_score_best.iterrows():
    best_model_name = model_spec['model_class']
    best_model_class = map_name_to_class[best_model_name]
    best_params = model_spec['parameter']
    
    # best_params['device'] = 'gpu'
    
    # Initialize the model with the best parameters
    model = best_model_class(**best_params)

    # Train the model on the entire training data
    model.fit(X_train_trans, y_train['target'])
    # save this for later use fir fitting a model on top of this
    y_val[f"{str(model).split('(')[0]}"] = model.predict(X_val_trans)
    # do predition for the test data
    y_test[f"{str(model).split('(')[0]}"] = model.predict(X_test_trans)
    

# %%
"""    
simple Ensembling methods are presented here:
"""
model = Lasso()
parameters = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]}
reg = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)

reg.fit(X=y_val.iloc[:,3:], y=y_val['target'])


y_test[f"Linear Ensemble"] = reg.best_estimator_.predict(y_test.iloc[:,3:])

print(y_test.head())

# %%
model_in_y_test = y_test.columns.to_list()[3:]
number_of_models = len(model_in_y_test)
y_true = np.tile(y_test['target'].values, (number_of_models, 1)).T  
# Extract y_pred (all columns starting from the 4th column)
y_pred = y_test.iloc[:, 3:].values

# Compute the mean squared error
mse = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')
mae = mean_squared_error(y_true=y_true, y_pred=y_pred, multioutput='raw_values')



df_performance = pd.DataFrame({'Algorithm':model_in_y_test,
             'MAE':mae,
             'MSE':mse})
metric_styled = df_performance.style.format({"MAE": "{:.4f}", 
                          "MSE": "{:.4f}", 
                        #   "MASE": "{:.3f}", 
                        #   "Forecast Bias": "{:.2f}%"}
                        }
                                           ).highlight_min(color='blue', subset=["MAE","MSE"])
print(metric_styled)

# %%
awful_model_to_not_show = ['Theta']
all_pred_df = pd.concat([y_val[['node', time_col, 'target']], y_test], axis=0)
all_pred_df.sort_values(['node', time_col], inplace=True)
all_node_names = all_pred_df['node'].unique()
random_nodes = np.random.choice(all_node_names, 4)
random_nodes
print(all_pred_df.head(n=6))


# Initialize subplots
fig = sp.make_subplots(rows=2, cols=2, subplot_titles=random_nodes)

# Loop through partitions
for i, node in enumerate(random_nodes):
    
    df_temp = all_pred_df[all_pred_df.node==node].iloc[:, 1:]
    df_temp.set_index(time_col, inplace=True)
    
    # print(df_temp.head())
    for j, model_name in enumerate(df_temp.columns):
        fig.add_trace( 
                        go.Scatter(
                                    y=df_temp[model_name].values,
                                    x=df_temp.index, name=model_name,
                                    # mode='lines+markers',
                                    line=dict(color=colors[j]), 
                                    # marker=dict(symbol=markers[j]),
                                    showlegend=(i == 0) 
                                    ), 
                        row=(i//2)+1, col=(i%2)+1)



fig.update_layout(
    title="Idle sample nodes and prediction",
    height=len(random_nodes) * 150  # Adjust height based on number of partitions
)

# print the Plotly figure
fig.show()

# %%
"""   
                    From heat map to heat map!
                    
To give a better presentation we show the prediction error for each node at each time step in the Heatmap format.

dark red ==> absoute value for predition error is 0, abs(y_hat(t) - y(t)) = 0
white  ==> 1 absoute value for predition error is 1, abs(y_hat(t) - y(t)) = 1

"""



best_model_index = df_performance.idxmin()['MSE']
best_model = df_performance.iloc[best_model_index,0]
y_test['error'] = (y_test['target'] - y_test[best_model]).abs()
print(f"Keep an eye on this value: {y_test['error'].max()}, if it is larger than 1 the figure is not correct" )


# pivot the table for plotting, call the function for getting longer intervals
df_stat_pivot = pd.pivot_table(y_test, index=time_col, columns=['node'], values=['error'])
df_stat_pivot = df_stat_pivot.droplevel(level=0, axis=1).copy()
df_stat_pivot.index = df_stat_pivot.index.strftime("%Y-%m-%d %H")
# df_stat_pivot.index = df_stat_pivot.index.strftime("%Y-%m-%d")


print(df_stat_pivot.head())

# plotting

# Define the ranges to plot and corresponding titles for each subplot
ranges = [(0, 120), (120, 275), (275, 475), (475, 675), (675, 875), (875, 1075), (1075, 1275), (1275, None)]

titles = [
    "fcn nodes",
    "gcn nodes",
    "hcn1-4 & tcn10-1175 nodes",
    "tcn1176-219 nodes",
   "tcn22-40 nodes",
   "tcn400-581 nodes",
   "tcn582-761 nodes",
   "tcn762-999 nodes"
]


map_time_column_names = {"time_30min_interval": "30 minutes interval",
                         "time_1hour_interval": "1 hour interval",
                         "time_3hour_interval": "3 hours interval",
                         "time_4hour_interval": "4 hours interval",
                         "time_6hour_interval": "6 hours interval",
                         "time_12hour_interval": "12 hours inteval",
                         "time_day_interval": "1 day interval"
                             }

# Create a subplot figure with 8 rows, 1 column
fig = sp.make_subplots(
    rows=8, cols=1,
    subplot_titles=titles,
    vertical_spacing=0.025
)

# Loop through each axis and range, adding heatmaps with a title for each
for i, (start, end) in enumerate(ranges):
    # Select the range of columns to plot
    sub_df = df_stat_pivot.iloc[:, start:end]
    
    # Add the heatmap to the subplot
    fig.add_trace(
        go.Heatmap(
            z=sub_df.values,
            x=sub_df.columns,
            y=sub_df.index,
            # colorscale='blackbody',
            # colorscale='electric',
                        colorscale='hot',
                # colorscale='inferno',


            zmin=0,
            zmax=1,
            colorbar=dict(
                title='Prediction error',
                orientation='v',
                len=0.25,  # Adjust length of colorbars
                yanchor='top',
                y=1 ,  # Offset to position the colorbar correctly
            ),
        ),
        row=i+1, col=1
    )


    # Ensure all ticks are printed
    fig.update_xaxes(
        tickmode="array",
        tickvals=sub_df.columns.tolist(),
        ticktext=sub_df.columns.tolist(),
        tickfont=dict(size=8),
        row=i+1, col=1
    )
    fig.update_yaxes(
        tickmode="array",
        tickvals=sub_df.index.tolist(),
        ticktext=sub_df.index.str[5:16].tolist(),
        tickfont=dict(size=8),
        row=i+1, col=1
    )
# Update the layout for better appearance
fig.update_layout(
    height=2600,  # Adjust height of the figure
    width=1400,  # Adjust width of the figure
    title=f"Absolute value Prediction error for - {map_time_column_names[time_col]}",
    title_x=0.5,
    showlegend=False
)

output_path_pdf = folder_path_saving_results / f"nodes_idle_partition_{time_col}.pdf"
output_path_png = folder_path_saving_results / f"nodes_idle_partition_{time_col}.png"
fig.write_image(output_path_pdf)
fig.write_image(output_path_png, scale=2)

# Show the interactive heatmap
fig.show()


# move the figures to the laptop for better visibility
# scp -r teimourh@snellius:/home/teimourh/slurm_energy_ml/results .



# %%
# to do:
"""
1) now we have a much cleaner pipline for preprocessing.
it would be way better to integrate the preprocessing with the models and change the paramter for the 
transformers as well
"""




