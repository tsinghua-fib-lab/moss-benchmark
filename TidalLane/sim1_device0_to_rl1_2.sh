#!/usr/bin/bash
set -x
set -e

# PART0_CITIES="china_beijing china_beijing_s us_newyork_s china_shanghai_s china_shanghai_c us_newyork_c"
PART0_CITIES="china_shanghai_s china_shanghai_c us_newyork_c"

echo "脚本已经开始将在3小时后继续执行。"

sleep 3h

echo "3小时已过脚本将继续执行。"

python3 run_ppo.py --data data/us_newyork_s --start 25200 --steps 10800 --exp us_newyork_s --device 2

for city in $PART0_CITIES; do
    
    python3 run_baseline.py --algo none --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_baseline.py --algo random --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_baseline.py --algo rule --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 2
done
