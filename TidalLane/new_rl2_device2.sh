#!/usr/bin/bash
set -x
set -e

echo "脚本已经开始将在7小时后继续执行。"

sleep 7h

echo "7小时已过脚本将继续执行。"

PART0_CITIES="france_paris_s"

for city in $PART0_CITIES; do
    for algo in none random rule; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_baseline.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 2
    done
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 2
done
