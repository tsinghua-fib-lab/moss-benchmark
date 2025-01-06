#!/usr/bin/bash
# MOSS
python3 run_moss.py
# Convert CityFlow data
python3 convertor/to_cityflow.py
# CityFlow
python3 run_cityflow.py
