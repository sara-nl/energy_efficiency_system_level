# %%
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import os
import time

from tqdm.notebook import tqdm





# %%
folder_path = Path('./system_analytics_2024/slurm_data')
file_path_parquet_reading_slurm= folder_path / 'slurm_data.parquet.gzip'
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)



# %%

df = pd.read_parquet(file_path_parquet_reading_slurm)
df.sample(n=5)

# %%

df['feature'] = df['feature'].str.split('\n')
df['length_of_feature'] = [len(l) for l in df['feature'].tolist()]
df['length_of_feature'].value_counts()[0:10]


# %%
""" 
I could not make the process that comee in the next cell as a multiprocessing unit.
It would be great to do so!
"""



# df_len = len(df)
# chunk_size = 500000

# list_1 = list(range(0, df_len, chunk_size))
# if list_1[-1] != df_len:
#     list_1.append(df_len)
    
# fin_list = list(zip(list_1[0:-1], list_1[1:]))




# def heavy_computation(index):
#     #print(len(df)) # this reurns the correct length of the dataframe
#     data_processed = []
#     for n in range(index[0], index[1]):
        
#         len_feature = int(df.iloc[n, :]['length_of_feature'])

#         if len_feature > 3:
#             job_id =int( df.iloc[n, :]['job_id'])
#             query_name = df.iloc[n, :]['feature'][0]
#             signal = df.iloc[n, :]['feature'][2:-1]
#             data = {'job_id': [job_id] * len(signal),
#                     'query_name': [query_name] * len(signal),
#                     'signal': signal}
#             # can we append the dictionary and later turn it into a data feame
#             data_processed.append(pd.DataFrame(data)) 
#     return data_processed
        



# t1 = time.perf_counter()
# # CPU-bound task: heavy computation
# max_workers = min(100, os.cpu_count())
# with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#     results = list(executor.map(heavy_computation, fin_list))




# t2 = time.perf_counter()

# print(f'Finished in {t2-t1} seconds')
# # Flatten the nested list of DataFrames
# flattened_results = [df for sublist in results for df in sublist]

# # Combine all processed chunks into a single DataFrame
# final_result = pd.concat(flattened_results, ignore_index=True)
    
    


# %%

# write a function to process the data row wise
lower_bound = 0
upper_bound = len(df)
data_processed = []

for n in range(lower_bound, upper_bound):
# df['length_of_feature']

    # len_feature = len(df.iloc[n, :]['feature'])
    len_feature = df.iloc[n, :]['length_of_feature']
    if len_feature > 3:
        job_id =int( df.iloc[n, :]['job_id'])
        query_name = df.iloc[n, :]['feature'][0]
        signal = df.iloc[n, :]['feature'][2:-1]
        
        
        data = {'job_id': [job_id] * len(signal),
                'query_name': [query_name] * len(signal),
                 'signal': signal}

        data_processed.append(pd.DataFrame(data))

df = pd.concat(data_processed, axis=0)
print(len(df))

# df.to_parquet(folder_path/'slurm_data_half_cleaned.parquet.gzip')



# %%
df['query_name'] = df['query_name'].str.split()
df['signal'] = df['signal'].str.split()
# df_temp.drop('feature', inplace=True, axis=1)
df.sample(n=5)

# %%

# get the length of signal name column
df['length_of_query'] = [len(l) for l in df['query_name'].tolist()]
df['length_of_signal'] = [len(l) for l in df['signal'].tolist()]
# sum(np.array(length_of_signal_name) - (np.array(length_of_values)))
print(df['length_of_query'].value_counts(),
df['length_of_signal'].value_counts())


# %%
# remove all strange signals
df = df[(df['length_of_signal']==11) | (df['length_of_signal']==13)]
df_11 = df[(df['length_of_signal']==11)].copy()
df_13 = df[(df['length_of_signal']==13)].copy()

# %%
# these are the signal names
signal_names = ['Submit', 'Eligible', 'Start', 'End', 'Elapsed',
                'JobID', 'JobName', 'State', 'AllocCPUS', 'TotalCPU',
                'AveRSS', 'MaxRSS',
                'NodeList']



# for the 13 signals
for i, signal_name in enumerate(signal_names):
    df_13[signal_name] = df_13['signal'].apply(lambda x:x[i])
    

# for the 11 signals
for i, signal_name in enumerate(signal_names[0:10] + signal_names[-1:]):
    df_11[signal_name] = df_11['signal'].apply(lambda x:x[i])


# concant the two frames:
df_cleaned = pd.concat([df_13, df_11], axis=0)
df_cleaned.sample(n=10)

# %%
df_cleaned.drop(['query_name','signal', 'length_of_query',
                 'length_of_signal', 'JobName'], axis=1, inplace=True)

df_cleaned.rename(columns={"JobID":"Slurm_job_id"}, inplace=True)

# %%
df_cleaned.to_parquet(folder_path / 'slurm_data_cleaned_final.parquet.gzip', compression='gzip')


