import pandas as pd
# the time interval that we put all the states in it.
map_time_col = {'time_1hour_interval':pd.Timedelta('1h'),
                'time_30min_interval':pd.Timedelta('30min'),
                'time_2hour_interval':pd.Timedelta('2h'),
                'time_3hour_interval':pd.Timedelta('3h'),
                'time_6hour_interval':pd.Timedelta('6h'),
                'time_12hour_interval':pd.Timedelta('12h'),
                }