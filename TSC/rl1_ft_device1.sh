#!/usr/bin/bash
set -x
set -e

PART0_CITIES="china_shanghai_s"

for city in $PART0_CITIES; do
    # for algo in ft mp ppo mplight efficient_mplight advanced_colight advanced_mplight colight frap; do
    for algo in ft mp ppo mplight; do
        python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
done
