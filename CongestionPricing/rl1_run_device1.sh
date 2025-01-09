#!/usr/bin/bash
set -x
set -e

PART_0_CITIES="us_newyork china_beijing_s china_shanghai_s"
PART_1_CITIES="france_paris_s us_newyork_s china_beijing_c"
PART_2_CITIES="china_shanghai_c france_paris_c us_newyork_c"

# ATTENTION: 临时设置--device
for city in $PART_1_CITIES; do
    for algo in none random deltatoll; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_all.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
    # train for 4 hours
    timeout 4h python3 run_all.py --algo eGCN --data data/$city --start 25200 --steps 10800 --exp $city --device 1  || true
done
