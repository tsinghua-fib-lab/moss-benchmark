This repo includes all the benchmarking code and hyper-parameters for the paper "A GPU-accelerated Large-scale Simulator for
Transportation System Optimization Benchmarking".

# Prerequisites
* Any linux distribution
* NVIDIA GPU with CUDA 12
* Python 3.9

# Setup
1. Download the dataset archive [data.tar.gz](https://fiblab-neurips2024-moss-benchmark.obs.cn-north-4.myhuaweicloud.com/data.tar.gz) and unpack it.
    ```bash
    tar xvf data.tar.gz
    ```
    If everything is correct, you should see the following structure:
    ```log
    data
    ├── congestion
    ├── dynamic
    ├── perf
    ├── road
    ├── tidal
    └── tsc
    ```
2. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
    In case of `'GLIBCXX_3.4.30' not found` error, also install:
    ```bash
    conda install -c conda-forge libstdcxx-ng=12
    ```

# Benchmarks
## Performance

Switch to the folder:
```bash
cd ./Perf
```

Install baseline simulators:
* Install `sumo`
    ```bash
    sudo add-apt-repository ppa:sumo/stable
    sudo apt-get update
    sudo apt-get install sumo sumo-tools sumo-doc
    pip install traci
    ```
* Install `cityflow`
    ```bash
    git clone --depth 1 https://github.com/cityflow-project/CityFlow
    pip install ./CityFlow
    ```
* Install `CBLab`
    ```bash
    git clone --depth 1 https://github.com/caradryanl/CityBrainLab
    pip install ./CityBrainLab/CBEngine/core
    ```

Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${city}/${algo}_${n}.json`, which contains a list that records the duration of each simulation step.

## Traffic Signal Control
Switch to the folder:
```bash
cd ./TSC
```

Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${algo}/${city}/${date}/info.log`, where each line records `ATT TP time`.

## Dynamic Lane Assignment
Switch to the folder:
```bash
cd ./DynamicLane
```

Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${algo}/${city}/${date}/info.log`, where each line records `ATT TP time`.

## Tidal Lane Control
Switch to the folder:
```bash
cd ./TidalLane
```

Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${algo}/${city}/${date}/info.log`, where each line records `ATT TP time`.

## Congestion Pricing
Switch to the folder:
```bash
cd ./CongestionPricing
```

Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${algo}/${city}/${date}/info.log`, where each line records `ATT TP time`.

## Road Planning

Switch to the folder:
```bash
cd ./CongestionPricing
```

### Step 1. Data Preparation

* Fetch the road net from OSM data of 2019.

    ```bash
    python preparation/fetch_2019_geojsons.py
    ```

* Find the difference between road network of 2019 and road network of 2024, mark those roads not in road network of 2019 as constructed in the past five years.

    ```bash
    for city in beijing shanghai newyork paris; do
        python preparation/fetch_candidate_way_ids.py --candidate_way_path ./data/${city}_candidate_ways.pkl --city $city
    done
    ```

    - `--candidate_way_path`: output path for all constructed ways from 2019 to 2024.
    - `--city`: name of the city to process.

* Select 50 roads out of constructed roads with the highest vehicle count during the simulation of morning peak and evening peak as the optimization set.

    ```bash
    for city in beijing shanghai newyork paris; do
        for condition in smooth normal congested; do
            python exp/select_50_optimize_way_ids.py \
                --opt_way_path ./data/${condition}_${city}_opt_ways.pkl \
                --city $city \
                --map_path ./data/moss.map_china_${city}.pb \
                --trip_path ./data/${condition}_${city}_trip.pb \
                --candidate_way_path ./data/${city}_candidate_ways.pkl
        done
    done
    ```

    - `--opt_way_path`: output path for 50 ways for algorithms to optimize.
    - `--city`: name of the city to process.
    - `--map_path`: map in `protobuf` format.
    - `--trip_path`: trip in `protobuf` format.
    - `--candidate_way_path`: output path of `fetch_candidate_way_ids.py`

### Step 2. Run Optimization Methods
Run all experiments:
```bash
bash run.sh
```
The results will be saved under `./log/${algo}/${city}/${date}/info.log`, where each line records `ATT TP time`.
