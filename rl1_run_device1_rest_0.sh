#!/usr/bin/bash
set -x
set -e

PART_1_CITIES="us_newyork_s china_beijing_c"

timeout 4h python3 run_all.py --algo eGCN --data data/france_paris_s --start 25200 --steps 10800 --exp france_paris_s --device 1

# ATTENTION: 临时设置--device
for city in $PART_1_CITIES; do
    for algo in none random deltatoll; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_all.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
    # train for 4 hours
    timeout 4h python3 run_all.py --algo eGCN --data data/$city --start 25200 --steps 10800 --exp $city --device 1
done
