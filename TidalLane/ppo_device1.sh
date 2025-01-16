#!/usr/bin/bash
set -x
set -e
# TODO:只差ppo的城市

PART0_CITIES="us_newyork us_newyork_s"

for city in $PART0_CITIES; do
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 1
done
