#!/usr/bin/bash
set -x
set -e

# normal traffic condition
NORMAL_CITIES="china_beijing china_shanghai france_paris us_newyork"
# smooth traffic condition
SMOOTH_CITIES="china_beijing_s china_shanghai_s france_paris_s us_newyork_s"
# congested traffic condition
CONGESTED_CITIES="china_beijing_c china_shanghai_c france_paris_c us_newyork_c"

for city in $NORMAL_CITIES $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in ft mp ppo mplight efficient_mplight advanced_colight advanced_mplight colight frap; do
        # train for 4 hours
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        timeout 4h python run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city
    done
done