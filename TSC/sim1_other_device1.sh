#!/usr/bin/bash
set -x
set -e

# smooth traffic condition
SMOOTH_CITIES="china_beijing_s china_shanghai_s france_paris_s us_newyork_s"
# congested traffic condition
CONGESTED_CITIES="china_beijing_c china_shanghai_c france_paris_c us_newyork_c"

# 其他拥堵条件
for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in mplight; do
         timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in efficient_mplight; do
         timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in advanced_mplight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in frap; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in colight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in advanced_colight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done


 