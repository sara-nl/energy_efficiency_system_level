#!/bin/bash

# we would like to get the system states every 15 seconds
for day in {1..2}
do
for i in {1..10}
do
        echo "############################" >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt   
        sleep 30  
done
done



#!/bin/bash

for day in {1..30}   
do
    for i in {1..1440}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt   
        sleep 60   
    done
done




#!/bin/bash

for day in {1..30}   
do
    for i in {1..720}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt   
        sleep 120   
    done
done
