#!/usr/bin/bash
set -x
set -e

PART_1_CITIES="us_newyork_s china_beijing_c"

python3 ../sleep_until_no_threads.py --command "python3 run_all.py --algo none --data data/france_paris_s"

python3 ../sleep_until_no_threads.py --command "python3 run_all.py --algo random --data data/france_paris_s"

python3 ../sleep_until_no_threads.py --command "python3 run_all.py --algo deltatoll --data data/france_paris_s"

python3 ../sleep_until_no_threads.py --command "python3 run_all.py --algo eGCN --data data/france_paris_s"

# ATTENTION: 临时设置--device
for city in $PART_1_CITIES; do
    for algo in none random deltatoll; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_all.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
    # train for 4 hours
    timeout 4h python3 run_all.py --algo eGCN --data data/$city --start 25200 --steps 10800 --exp $city --device 1  || true
done
