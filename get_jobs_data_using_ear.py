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
      
      
      
      # Here I calculate the min time for an id and step id
      df_start = pd.DataFrame(df.groupby('job_id')['start_time'].min()).reset_index()
      df_end = pd.DataFrame(df.groupby('job_id')['end_time'].max()).reset_index()
      df_time = pd.merge(left=df_start, right=df_end)

      df_time.rename(columns={'start_time':'job_start_time', 
                              'end_time':'job_end_time'}, inplace=True)
      df = pd.merge(df, df_time, how='inner', on='job_id').copy()
      df['job_start_time_date'] = df['job_start_time'].dt.date

      return df
