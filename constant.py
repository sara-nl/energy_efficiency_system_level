import pandas as pd
# the time interval that we put all the states in it.
MAP_TIME_COL = {'time_1hour_interval':pd.Timedelta('1h'),
                'time_30min_interval':pd.Timedelta('30min'),
                'time_2hour_interval':pd.Timedelta('2h'),
                'time_3hour_interval':pd.Timedelta('3h'),
                'time_4hour_interval':pd.Timedelta('4h'),
                'time_6hour_interval':pd.Timedelta('6h'),
                'time_12hour_interval':pd.Timedelta('12h'),
                'time_day_interval':pd.Timedelta('1d'),
                'time_week_interval':pd.Timedelta('7d')
                }



PARTITION_NAME_TO_NODE = { 'NVIDIA_A100': ["gcn" + str(i) for i in range(4, 73)] + ["gcn2", "gcn3"],
                            'NVIDIA_H100': ["gcn" + str(i) for i in range(73, 161)],
                            # this node list is taken from /etc/slurm/slurm.conf
                            'AMD_ROME': ["tcn" + str(i) for i in range(4, 526)],
                            'AMD_GENOA': ["tcn" + str(i) for i in range(527, 1311)],
                            # this node list is taken from /etc/slurm/slurm.conf
                            'himem_4tb': ['hcn1', 'hcn2'],
                            'himem_8tb': ['hcn3', 'hcn4'],
                            # we don't have specific parition for fcn nodes, this is just the names
                            'fcn_nodes' : ['fcn' + str(i) for i in range(1, 121)]
                            }

NODE_TO_PARTITION_NAME = {
    node: par_name
    for par_name, node_list in PARTITION_NAME_TO_NODE.items()
    for node in node_list
}


NODE_TO_PARTITION_NAME_SHORT = {
    node: 'himem' if par_name in {'himem_4tb', 'himem_8tb'} else par_name
    for par_name, node_list in PARTITION_NAME_TO_NODE.items()
    for node in node_list
}



# price for electricity per kilo wat hour
ELEC_PRICE_KWH = 0.30
# CO2 emmission kg per kWh, we use the data from
# reference: https://www.cbs.nl/nl-nl/achtergrond/2023/51/rendementen-en-co2-emissie-van-elektriciteitsproductie-in-nederland-update-2022
CO2_EMISSION = 0.27

KILO_WAT_CONVERSION = 1000