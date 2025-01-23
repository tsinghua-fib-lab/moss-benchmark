#!/usr/bin/bash
set -x
set -e

sleep 11h
# terminated exps
timeout 4h python3 run_efficient_mplight.py --data ./data/china_shanghai_s --start 25200 --steps 10800 --exp china_shanghai_s --device 1 || true