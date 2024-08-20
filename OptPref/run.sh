#!/usr/bin/bash
# MOSS
python run_moss.py
# Convert CityFlow data
python convertor/to_cityflow.py
# CityFlow
python run_cityflow.py