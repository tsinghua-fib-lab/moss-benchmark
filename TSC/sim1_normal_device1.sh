#!/usr/bin/bash
set -x
set -e

# normal traffic condition
NORMAL_CITIES="china_beijing china_shanghai france_paris us_newyork"

# 正常城市的三个实验TSC部分

for city in $NORMAL_CITIES; do
    for algo in mplight; do
         timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $NORMAL_CITIES; do
    for algo in efficient_mplight; do
         timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $NORMAL_CITIES; do
    for algo in advanced_mplight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done
