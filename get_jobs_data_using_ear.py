# %%
import pandas as pd
from pathlib import Path




folder_path_ear_data = Path('/projects/2/prjs1098/system_analytics_2024/ear_data')

file_path_parquet_reading_jobs = folder_path_ear_data / 'jobs.parquet.gzip'
file_path_parquet_reading_apps = folder_path_ear_data / 'applications.parquet.gzip'

def get_jobs_data(job_table_path, apps_table_path):
      df_jobs = pd.read_parquet(file_path_parquet_reading_jobs)
      df_apps = pd.read_parquet(file_path_parquet_reading_apps)

      # merget the data frames
      df = pd.merge(left=df_jobs, right=df_apps,
                  left_on=['id', 'step_id'], right_on=['job_id', 'step_id'])
      # drop the id column since we have the job_id
      df.drop('id', axis=1, inplace=True)
      df.drop('app_id', axis=1, inplace=True, errors='ignore')
      df['node_id'] = df['node_id'].str.split("\x00").str[0]
      # change the time to pd date time for better readability
      cols = ['start_time', 'end_time', 'start_mpi_time', 'end_mpi_time']
      df[cols] = df[cols].apply(lambda x: pd.to_datetime(x, unit='s'))

      return df
