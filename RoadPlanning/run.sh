#!/usr/bin/bash
set -x
set -e

for city in beijing shanghai newyork paris; do
    for condition in smooth normal congest; do
        args="--city $city --map_path ./data/moss.map_${city}.pb --trip_path ./data/${condition}_${city}_trip.pb --opt_way_path ./data/${condition}_${city}_opt_ways.pkl --condition $condition"
        # NoChange
        python3 exp/run.py --algo none   --epochs 1  $args
        # Random
        python3 exp/run.py --algo random --epochs 5  $args
        # Rule
        python3 exp/run.py --algo rule   --epochs 1  $args
        # GeneralBO
        python3 exp/run.py --algo bo     --epochs 20 $args
        # Simulated annealing
        python3 exp/run_sa.py                             $args
    done
done
