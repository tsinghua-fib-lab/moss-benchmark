"""
generate opt_way_ids
"""

import argparse
import logging
import os
import pickle
import subprocess

import numpy as np
from mosstool.type import Map
from utils.const import *
from utils.utils import get_geojson_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_id", help="cuda device id", type=int, default=0)
    parser.add_argument("--opt_way_path", default="")
    parser.add_argument("--city", default="")
    parser.add_argument("--candidate_way_path", default="")
    parser.add_argument("--map_path", default="")
    parser.add_argument("--trip_path", default="")
    return parser.parse_args()


args = get_args()
DEVICE_ID = args.device_id
CITY = args.city
OPT_WAY_PATH = args.opt_way_path
ROADNET_PATH = f"./data/roadnet_{CITY}.geojson"
ALL_CANDIDATE_WAY_PATH = args.candidate_way_path
ORIG_MAP_PATH = args.map_path
ORIG_TRIP_PATH = args.trip_path
with open(ORIG_MAP_PATH, "rb") as f:
    ORIG_MAP = Map()
    ORIG_MAP.ParseFromString(f.read())
(orig_topo_dict, way_id2junc_id, way_id2line, way_id2idx) = get_geojson_data(
    roadnet_path=ROADNET_PATH, proj_str=ORIG_MAP.header.projection
)
# from 2019 osm data
if not os.path.exists("./LOOP_DATA/"):
    os.makedirs("./LOOP_DATA/", exist_ok=True)
all_build_way_ids = pickle.load(open(ALL_CANDIDATE_WAY_PATH, "rb"))
loop_v_cnts_path = f"./LOOP_DATA/BUILD_{CITY}_ave_v_cnts.pkl"
# morning peak
cmd = f"python3 exp/utils/dump_v_cnts.py --map_path {ORIG_MAP_PATH} --agent_path {ORIG_TRIP_PATH} --output_path {loop_v_cnts_path} --device_id {DEVICE_ID}".split(
    " "
)
subprocess.run(cmd, cwd="./", check=True)
ave_v_cnts_morning = pickle.load(open(loop_v_cnts_path, "rb"))
# evening peak
cmd = f"python3 exp/utils/dump_v_cnts.py --map_path {ORIG_MAP_PATH} --start_step {17*3600} --agent_path {ORIG_TRIP_PATH} --output_path {loop_v_cnts_path} --device_id {DEVICE_ID}".split(
    " "
)
subprocess.run(cmd, cwd="./", check=True)
ave_v_cnts_night = pickle.load(open(loop_v_cnts_path, "rb"))
ave_v_cnts = np.mean([ave_v_cnts_night, ave_v_cnts_morning], axis=0)


def way_id2v_cnt(way_id):
    idx = way_id2idx[way_id]
    v_cnt = ave_v_cnts[idx]
    return v_cnt


opt_way_ids = sorted(all_build_way_ids, key=lambda x: way_id2v_cnt(x))[
    -NUM_OF_BUILD_WAYS:
]
pickle.dump(opt_way_ids, open(OPT_WAY_PATH, "wb"))
