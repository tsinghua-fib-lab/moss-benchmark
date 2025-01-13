#!/usr/bin/bash
set -x
set -e

PART0_CITIES="china_beijing china_beijing_s china_beijing_c"

for city in $PART0_CITIES; do
    # for algo in ft mp ppo mplight efficient_mplight advanced_colight advanced_mplight colight frap; do
    for algo in colight frap; do
        python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 0
    done
done
