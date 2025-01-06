#!/usr/bin/bash
set -x
set -e
# Fetch the road net from OSM data of 2019.
python3 preparation/fetch_2019_geojsons.py
# Find the difference between road network of 2019 and road network of 2024, mark those roads not in road network of 2019 as constructed in the past five years.
for city in beijing shanghai newyork paris; do
    python3 preparation/fetch_candidate_way_ids.py --candidate_way_path ./data/${city}_candidate_ways.pkl --city $city
done
# Select 50 roads out of constructed roads with the highest vehicle count during the simulation of morning peak and evening peak as the optimization set.
for city in beijing shanghai newyork paris; do
        for condition in smooth normal congested; do
            python3 exp/select_50_optimize_way_ids.py \
                --opt_way_path ./data/${condition}_${city}_opt_ways.pkl \
                --city $city \
                --map_path ./data/moss.map_${city}.pb \
                --trip_path ./data/${condition}_${city}_trip.pb \
                --candidate_way_path ./data/${city}_candidate_ways.pkl
        done
done
