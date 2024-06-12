#!/usr/bin/bash
set -x
set -e

for city in beijing shanghai newyork paris; do
    for condition in smooth normal congested; do
        args="--city $city --map_path ./data/moss.map_${city}.pb --trip_path ./data/${condition}_${city}_trip.pb --opt_way_path ./data/${condition}_${city}_opt_ways.pkl"
        # NoChange
        python exp/run.py --iter_type none   --epochs 1  $args
        # Random
        python exp/run.py --iter_type random --epochs 5  $args
        # Rule
        python exp/run.py --iter_type rule   --epochs 1  $args
        # GeneralBO
        python exp/run.py --iter_type bo     --epochs 20 $args
        # Simulated annealing
        python exp/run_sa.py                             $args
    done
done
