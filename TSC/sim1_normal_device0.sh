#!/usr/bin/bash
set -x
set -e

# normal traffic condition
UNDONE_NORMAL_CITIES="china_shanghai france_paris"


for city in $UNDONE_NORMAL_CITIES; do
    for algo in advanced_mplight; do
        timeout 4h python3 run_$algo.py --data ./data/$city --start 25200 --steps 10800 --exp $city --device 0 || true
    done
done
