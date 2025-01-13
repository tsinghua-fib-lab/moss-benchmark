#!/usr/bin/bash
set -x
set -e

PART1_CITIES="france_paris_c"

for city in $PART1_CITIES; do
    # for algo in ft mp ppo mplight efficient_mplight advanced_colight advanced_mplight colight frap; do
    for algo in ft mp ppo mplight; do
        # train for 4 hours
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 2
    done
done
