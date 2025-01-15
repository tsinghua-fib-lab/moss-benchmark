#!/usr/bin/bash
set -x
set -e

# 这个是没跑完colight的城市
UNDONE_NORMAL_CITIES="france_paris us_newyork"

# 这些是还没跑完efficient_mplight的city
# smooth traffic condition
UNDONE_SMOOTH_CITIES="china_beijing_s china_shanghai_s us_newyork_s"
# congested traffic condition
UNDONE_CONGESTED_CITIES="china_beijing_c france_paris_c us_newyork_c"

# normal traffic condition
NORMAL_CITIES="china_beijing china_shanghai france_paris us_newyork"
# smooth traffic condition
SMOOTH_CITIES="china_beijing_s china_shanghai_s france_paris_s us_newyork_s"
# congested traffic condition
CONGESTED_CITIES="china_beijing_c china_shanghai_c france_paris_c us_newyork_c"


for city in $UNDONE_NORMAL_CITIES; do
    for algo in colight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $NORMAL_CITIES; do
    for algo in advanced_mplight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $NORMAL_CITIES; do
    for algo in advanced_colight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

# 其他拥堵条件
for city in $UNDONE_SMOOTH_CITIES $UNDONE_CONGESTED_CITIES; do
    for algo in efficient_mplight; do
        python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1
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
    for algo in advanced_mplight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done

for city in $SMOOTH_CITIES $CONGESTED_CITIES; do
    for algo in advanced_colight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 1 || true
    done
done


 