#!/usr/bin/bash
set -x
set -e

PART0_CITIES="china_beijing china_beijing_s china_shanghai_s"

for city in $PART0_CITIES; do
    for algo in none random rule; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_baseline.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 1
done
