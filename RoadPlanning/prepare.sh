#!/usr/bin/bash
set -x
set -e
python3 preparation/fetch_2019_geojsons.py
for city in beijing shanghai newyork paris; do
    python3 preparation/fetch_candidate_way_ids.py --candidate_way_path ./data/${city}_candidate_ways.pkl --city $city
done
for city in beijing shanghai newyork paris; do
        for condition in smooth normal congested; do
            python3 exp/select_50_optimize_way_ids.py \
                --opt_way_path ./data/${condition}_${city}_opt_ways.pkl \
                --city $city \
                --map_path ./data/moss.map_china_${city}.pb \
                --trip_path ./data/${condition}_${city}_trip.pb \
                --candidate_way_path ./data/${city}_candidate_ways.pkl
        done
done
