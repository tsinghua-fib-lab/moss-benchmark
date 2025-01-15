#!/usr/bin/bash
set -x
set -e

# ATTENTION:rl1寄了 因此用rl2跑完剩余的

python3 run_ppo.py --data data/france_paris_s --start 25200 --steps 10800 --exp france_paris_s --device 1

PART0_CITIES="china_shanghai_s"

for city in $PART0_CITIES; do
    for algo in none random rule; do
        # simulate from 7:00 to 10:00, i.e. start at 7*3600=25200 seconds and simulet 3*3600=10800 steps
        python3 run_baseline.py --algo $algo --data data/$city --start 25200 --steps 10800 --exp $city --device 1
    done
    python3 run_ppo.py --data data/$city --start 25200 --steps 10800 --exp $city --device 1
done
