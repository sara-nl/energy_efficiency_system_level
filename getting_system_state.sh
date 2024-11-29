



#!/bin/bash

for day in {16..30}   
do
    for i in {1..488}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt   
        sleep 180   
    done
done




#!/bin/bash

for day in {13..30}   
do
    for i in {1..480}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt   
        sleep 180   
    done
done



# change the resolution to 2 minutes

for day in {17..30}   
do
    for i in {1..720}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt   
        sleep 120   
    done
done