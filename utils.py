
import re
import pandas as pd
import subprocess



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



    return df_stat
