#!/usr/bin/bash
set -x
set -e

PART0_CITIES="china_shanghai_c"

for city in $PART0_CITIES; do
    
    python3 run_baseline.py --algo none --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_baseline.py --algo random --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_baseline.py --algo rule --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 2
done
