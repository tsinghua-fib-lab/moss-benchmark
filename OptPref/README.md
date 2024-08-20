# Optimization Preferences

In this scenario, there are three steps to run our experiment. 

## Step 1. Apply the Code Modifications on `CityFlow`

1. Clone the original `CityFlow` source codes.
   - `git clone https://github.com/cityflow-project/CityFlow.git`

2. Apply the patch provided in folder `data`.
   - `git apply ./diff_data/diff_cityflow.patch`

## Step 2. Convert Our Data to `CityFlow` Format

1. Assign file paths for both input and output `map`, `agent` file paths.
   -  `convertor/to_cityflow.py`

## Step 3. Run Optimization Method on engines

1. Run `CoLight` algorithm with `CityFlow`.
   - `run_cityflow.py`
2. Run `CoLight` algorithm with `MOSS`.
   - `run_moss.py`