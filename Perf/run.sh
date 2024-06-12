#!/usr/bin/bash
set -x
set -e

CITIES="Changchun Hangzhou Jinan Nanchang Shanghai Shenzhen eff_10inter eff_1e2inter eff_1e2veh eff_1e3inter eff_1e3veh eff_1e4inter_Shanghai eff_1e4veh eff_1e5veh eff_1e6veh_1e4.5inter"

# repeat 3 times
for n in 0 1 2; do
    for data in $CITIES; do
        for sim in cityflow cblab sumo moss; do
            python run_$sim.py --data $data --output log/$data/${sim}_$n.json
        done
    done
done