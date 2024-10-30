# %%
import mysql.connector
import pandas as pd
import pickle
from pathlib import Path
import subprocess
from tqdm import tqdm

# %%
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# %%
folder_path = Path('./data_storage')
# db_path = folder_path / 'metrics_stats_1.db'
file_path_parquet = folder_path / 'jobs.parquet.gzip'
file_path_parquet_writing = folder_path / 'slurm_ear_second_half.parquet.gzip'

# %%
df = pd.read_parquet(file_path_parquet)
df.head()

# %%
job_ids = df['id'].unique()

print(f"How many jobs: {len(job_ids)}")

first_half = int(0.90 * len(job_ids)//2)
print(f"Half of the jobs: {first_half}")

# save some memory
del df
# %%
slurm_job_data = {}


for job_id in tqdm(job_ids[first_half:-1]):
    # Run the 'sacct' command with job ID and format options
    command = ['sacct', '-j', str(job_id), '--format=Submit,Eligible,Start,End,Elapsed,JobID,JobName,State,AllocCPUs,TotalCPU,AveRSS,MaxRSS,NodeList']
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.stderr:
        print("Standard Error:\n", result.stderr)
    else:
        slurm_job_data[job_id] = result.stdout
    
    


# %%
df = pd.DataFrame(pd.Series(slurm_job_data))
df.to_parquet(file_path_parquet_writing, compression='gzip')



