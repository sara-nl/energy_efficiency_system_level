



#!/bin/bash

for day in {57..90}   
do
    for i in {1..488}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states/system_states_${day}.txt   
        sleep 180   
    done
done




#!/bin/bash

for day in {65..90}   
do
    for i in {1..960}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int4/system_states_${day}.txt   
        sleep 180   
    done
done



# change the resolution to 2 minutes

for day in {63..90}   
do
    for i in {1..1440}  
    do 
        echo "############################" >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt
        sinfo -l >> system_analytics_2024/slurm_data/system_states_int5/system_states_${day}.txt   
        sleep 120   
    done
done