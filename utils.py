
import re
import pandas as pd
import subprocess


from constant import KILO_WAT_CONVERSION, ELEC_PRICE_KWH, CO2_EMISSION, MAP_TIME_COL, NODE_TO_PARTITION_NAME


def format_node_names(node_name: str) -> str: 
    """This function cleans the node names that we have
    obtained from slurm data base. 
    for example [tcn97, tcn99-tcn101] ==> tcn97, tcn99, tcn100, tcn101

    Args:
        node_name (str): _it is a string that we want to process_

    Returns:
        _string_: 
    """
    # Check if the node name contains brackets (indicating range or list format)
    if '[' in node_name:
        # Extract prefix and the numbers part
        prefix, nums = re.match(r"(\w+)\[(.+)\]", node_name).groups()
        
        # Split on comma and process each item
        formatted_nodes = []
        for part in nums.split(','):
            part = part.strip()
            if '-' in part:  # It's a range
                start, end = map(int, part.split('-'))
                formatted_nodes.extend([f"{prefix}{i}" for i in range(start, end + 1)])
            else:  # It's a single number
                formatted_nodes.append(f"{prefix}{part}")
        
        return f"{','.join(formatted_nodes)}"
    else:
        # No brackets, treat as a single node name
        return f"{node_name}"
    
    
    
    
def get_list(file_path):
    """Reads a file and returns its contents as a list of lines.

    Args:
        file_path (str): The path to the file to read.

    Returns:
        list of str: A list where each element is a line from the file.
    """
    names = []
    with open(file_path, "r") as file:
        for line in file:
            # Strip leading/trailing whitespace and check if line is a comment
            line = line.strip()
            if not line.startswith("#") and line:  # If the line is not a comment and is not empty
                names.append(line.split('.')[0])

    return names





       

def get_slurm_data(job_ids: list[int]) -> object:
    """a function for getting data from slurm

    Args:
        job_ids (list[int]): list of job ids

    Returns:
        object: a pandas dataframe
    """

    slurm_job_data = {}

    for job_id in job_ids:
        # Run the 'sacct' command with job ID and format options
        command = ['sacct', '-j', str(job_id),'--parsable',
        '--format=Submit,Eligible,Start,End,Elapsed,JobID,JobName,State,AllocCPUs,TotalCPU,NodeList']
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.stderr:
            print("Standard Error:\n", result.stderr)
        else:
            slurm_job_data[job_id] = result.stdout
        

    df = pd.DataFrame(pd.Series(slurm_job_data))
    df.index.name = 'job_id'
    df.reset_index(inplace=True)
    df.rename(columns= {0: 'feature'}, inplace=True)
    return df



def preprocess_slurm(df: pd.DataFrame) -> pd.DataFrame:
    """processing the slurm data

    Args:
        df (object): a data frame with job id and the feature

    Returns:
        object: a dataframe with extracted features as its columns
    """

    df['feature'] = df['feature'].str.split('\n')
    df['length_of_feature'] = [len(l) for l in df['feature'].tolist()]
    # df['length_of_feature'].value_counts()[0:10]
    
    
    
 
    lower_bound = 0
    upper_bound = len(df)
    data_processed = []

    for n in range(lower_bound, upper_bound):

        len_feature = df.iloc[n, :]['length_of_feature']
        if len_feature > 2:
            job_id =int( df.iloc[n, :]['job_id'])
            query_name = df.iloc[n, :]['feature'][0]
            signal = df.iloc[n, :]['feature'][1:-1]
            
            
            data = {'job_id': [job_id] * len(signal),
                    'query_name': [query_name] * len(signal),
                    'signal': signal}

            data_processed.append(pd.DataFrame(data))

    df = pd.concat(data_processed, axis=0)
    df['query_name'] = df['query_name'].str.split('|')
    df['signal'] = df['signal'].str.split('|')
    # get the length of signal name column
    df['length_of_query'] = [len(l) for l in df['query_name'].tolist()]
    df['length_of_signal'] = [len(l) for l in df['signal'].tolist()]
    
    


    signal_names = df['query_name'].iloc[0][0:-1]
        # for the 13 signals
    for i, signal_name in enumerate(signal_names):
        df[signal_name] = df['signal'].apply(lambda x:x[i])
        
        
        
    df['formatted_node_names'] = df['NodeList'].apply(format_node_names)
    df.drop(['query_name','signal', 'length_of_query',
                 'length_of_signal', 'JobName'], axis=1, inplace=True)

    df.rename(columns={"JobID":"Slurm_job_id"}, inplace=True)

 
    df.sort_values(by='job_id', inplace=True)


    return df




def get_idle_proportion(df: pd.DataFrame, time_col: str) -> pd.DataFrame:

    df_with_duration = df.copy()
    # how long does a state last? we are ignoreing those samples outside of the time interval
    df_with_duration['state_duration'] = df_with_duration.groupby(['node', time_col])['time'].diff(1).shift(-1)
    # drop the rows that beccome Nan due to shift
    df_with_duration = df_with_duration[~(df_with_duration['state_duration'].isna())]
    
    # compute the total time a node was in a state in a given inteval
    df_temp = df_with_duration.groupby(['node', time_col, 'state' ], as_index=False)[['state_duration']].sum()
    df_temp.sort_values(['node', time_col], inplace=True)
    df_temp.rename(columns={'state_duration':'state_duration_in_interval'}, inplace=True)


    # get the total time for all the states
    df_total = df_temp.groupby(['node', time_col], as_index=False)['state_duration_in_interval'].sum().copy()
    df_total.rename(columns={'state_duration_in_interval':'all_state_durations_in_interval'}, inplace=True)

    # get the idle time for the states
    df_idle = df_temp[(df_temp['state']=='idle')].copy()
    df_idle.drop(columns='state', inplace=True)
    df_idle.rename(columns={'state_duration_in_interval':'idle_duration'}, inplace=True)

    # join the two data frame based on node and time
    df_stat = pd.merge(df_idle, df_total, how='outer', on=['node', time_col])
    df_stat.fillna(value=pd.Timedelta('0s'), inplace=True)
    df_stat.sort_values(['node', time_col], inplace=True)

    df_stat['idle_proportion'] = (df_stat['idle_duration'] / df_stat['all_state_durations_in_interval'])



    return df_stat, df_idle, df_total





def get_prom_average_node_sinfo(prom_file_paths, time_interval, time_interval_name):

    # change this fucntion to only accept path and granularity for getting the average
    
    df_average_list = []
    
    for file_path in prom_file_paths:
        # read a dataframe
        df = pd.read_parquet(file_path)
        # sort based on node and timestemp
        df.sort_values(['node', 'timestamp'], inplace=True)
        # drop duplicates
        df.drop_duplicates(['node', 'timestamp'], inplace=True)

        # turn the time to pandas time
        df['time'] = pd.to_datetime(df['timestamp'], unit='s')
        # add the interval tag
        df[time_interval_name] = ((df['time'] )).dt.floor(freq=time_interval)
        
        # average measuremnt
        df_prom_average = df.groupby(['node', time_interval_name], as_index=False).mean(numeric_only=True).copy()
        # drop the time stamps
        df_prom_average.drop(columns=['timestamp'], inplace=True)
        
        # save memory
        del df
        # if there is only one file then:
        if len(prom_file_paths)>1:
            df_average_list.append(df_prom_average)
            del df_prom_average
        else:
            df_prom_average.drop_duplicates(['node', time_interval_name], inplace=True)
            return df_prom_average
    # put all the averages in a single dataframe
    df_prom_average_all = pd.concat(df_average_list, axis=0)
    # sort based on node and time interval name
    df_prom_average_all.sort_values(['node', time_interval_name], inplace=True)
    # do antoher group by to ensure that if there is overlap between the data then we get things right
    df_prom_average = df_prom_average_all.groupby(['node', time_interval_name], 
                                                  as_index=False).mean(numeric_only=True).copy()

    
    df_prom_average.drop_duplicates(['node', time_interval_name], inplace=True)
    return df_prom_average
        




def get_jobs_data(job_table_path, apps_table_path):
      df_jobs = pd.read_parquet(job_table_path)
      df_apps = pd.read_parquet(apps_table_path)

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
  
  
  
  
  
def energy_motivation_rigourous(df: pd.DataFrame, df_idle_power_average: pd.DataFrame,
                                time_col: str)-> tuple[pd.DataFrame, dict]:

    
    df_stat_highly_idle, _, _ = get_idle_proportion(df, time_col)


    # turn the idle duration to hour
    df_stat_highly_idle['idle_duration_hour'] = (df_stat_highly_idle['idle_duration'].dt.total_seconds()/3600)
    
    df_high_idle_with_average_power = pd.merge(left=df_stat_highly_idle, right=df_idle_power_average, 
              how='left', on='node')

    # use surf sys power and compute the idle kilo wat hour
    df_high_idle_with_average_power['idle_kilo_watt_hour'] = (df_high_idle_with_average_power['idle_duration_hour']
                                                            .multiply(df_high_idle_with_average_power['surf_sys_power_mean']
                                                                    /KILO_WAT_CONVERSION, fill_value=0))
        # compute the price for the kilo-wat hour
    df_high_idle_with_average_power['financial_cost'] = (df_high_idle_with_average_power['idle_kilo_watt_hour'] 
                                                                                            * ELEC_PRICE_KWH)
    # add all the cost together
    total_financial_cost = df_high_idle_with_average_power['financial_cost'].sum()

    # compute Co2 emission for kilo-watt hour
    df_high_idle_with_average_power['co2_emission'] = (df_high_idle_with_average_power['idle_kilo_watt_hour'] 
                                                                                            * CO2_EMISSION)
    # add all the emssion together
    total_co2_emission = df_high_idle_with_average_power['co2_emission'].sum()

    # add all the idle duration and energy usages together
    total_year_idle = (df_high_idle_with_average_power['idle_duration_hour'].sum())/ (24 * 365)
    total_power_idle = (df_high_idle_with_average_power['idle_kilo_watt_hour'].sum())
    
    df_temp = df_high_idle_with_average_power[['node', 'idle_duration_hour','idle_kilo_watt_hour',
                                           'financial_cost', 'co2_emission']].copy()
    # df_temp['normalized_financial_cost'] = df_temp['financial_cost']/total_financial_cost
    # df_temp['normalized_co2_emission'] = df_temp['co2_emission']/total_co2_emission
    df_temp['partition'] = df_temp['node'].apply(lambda x: NODE_TO_PARTITION_NAME.get(x, 'other'))
    # df_stat_fin_co2_cost = df_temp.groupby('node_type', as_index=False)[['normalized_financial_cost', 
    #                                                                      'normalized_co2_emission']].sum()


    df_stat_fin_co2_cost = df_temp.groupby('partition', as_index=False)[['idle_duration_hour', 'idle_kilo_watt_hour',
                                                                        'financial_cost','co2_emission']].sum()
    total_data = {"total_year_idle":total_year_idle,
                  "total_power_idle":total_power_idle,
                  "total_financial_cost":total_financial_cost,
                  "total_co2_emission": total_co2_emission}
    
    return df_stat_fin_co2_cost, total_data