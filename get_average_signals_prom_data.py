import pandas as pd
from pathlib import Path
import time

from constant import MAP_TIME_COL
from utils import get_prom_average_node_sinfo


folder_path_prom_data = Path('./system_analytics_2024/prom_data')
all_prom_file_paths = list(folder_path_prom_data.glob("*.gzip"))
# select the granularity for getting average
time_col = 'time_1hour_interval'
time_formated = pd.Timestamp(time.time(), unit='s').strftime('%Y-%m-%d')



df_prom_average = get_prom_average_node_sinfo(all_prom_file_paths, MAP_TIME_COL[time_col], time_col)
print('process is done')
df_prom_average.rename(columns={'time_1hour_interval':'time'}, inplace=True)
df_prom_average.to_parquet(folder_path_prom_data/'average_signal_prom'/
                           f"prom_average_data_{time_formated}.parquet.gzip", compression='gzip')
print('writing is done')
