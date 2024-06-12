import argparse
import logging
import pickle
from math import ceil
from multiprocessing import cpu_count

import geojson
import numpy as np
import pyproj
from const import *
from geojson import FeatureCollection
from shapely.geometry import LineString
from tqdm.contrib.concurrent import process_map


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_way_path", default="")
    parser.add_argument("--city", default="")
    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
args = get_args()
CITY = args.city
ALL_CANDIDATE_WAY_PATH = args.candidate_way_path
HIGHWAY_TO_SCORE = {highway: ii for ii, highway in enumerate(FILTER_HIGHWAYS)}


def is_ok_feature(feature, line, min_road_length):
    if line.length < min_road_length:
        return False
    prop = feature["properties"]
    if prop["highway"] not in {
        "motorway",
        "trunk",
        "primary",
    }:
        return False
    return True


def match_unit(arg):
    global mid_points_2019, all_2019_lines, net_features_2019, point2gird
    feature, line, point_2024 = arg
    if not is_ok_feature(feature, line, MIN_ROAD_LENGTH):
        return None
    MATCHED = False
    f_highway = feature["properties"]["highway"]
    x1, y1 = point_2024.x, point_2024.y
    grid_2024 = point2gird[point_2024]
    for f_2019, line_2019, point_2019 in zip(
        net_features_2019, all_2019_lines, mid_points_2019
    ):
        if MATCHED:
            return None
        grid_2019 = point2gird[point_2019]
        if not grid_2019 == grid_2024:
            continue
        x2, y2 = point_2019.x, point_2019.y
        f_highway_2019 = f_2019["properties"]["highway"]
        if (
            np.abs(HIGHWAY_TO_SCORE[f_highway_2019] - HIGHWAY_TO_SCORE[f_highway])
            > HIGHWAY_GATE
        ):
            continue
        if np.sqrt(2) * (np.abs(x1 - x2) + abs(y1 - y2)) - line_2019.length < DIS_2GATE:
            if line_2019.distance(line) > DIS_GATE:
                continue
            if point_2024.distance(line_2019) < 1.2 * DIS_GATE:
                MATCHED = True
    if not MATCHED:
        return feature
    else:
        return None


city_args = {
    "beijing": (2, 15, 500),
    "shanghai": (2, 15, 500),
    "newyork": (1, 5, 200),
    "paris": (1, 5, 200),
}
HIGHWAY_GATE, DIS_GATE, MIN_ROAD_LENGTH = city_args[CITY]
bbox = ALL_BBOX[CITY]
DIS_2GATE = 2 * DIS_GATE
proj_str = f"+proj=tmerc +lat_0={(bbox['max_lat'] + bbox['min_lat']) / 2} +lon_0={(bbox['max_lon'] + bbox['min_lon']) / 2}"
projector = pyproj.Proj(proj_str)
path_2019 = f"./data/OSM_2019_roadnet_{CITY}.geojson"
path_2024 = f"./data/roadnet_{CITY}.geojson"
with open(path_2019, "r") as f:
    net_2019 = geojson.load(f)
with open(path_2024, "r") as f:
    net_2024 = geojson.load(f)
diff_features = []
global mid_points_2019, all_2019_lines, net_features_2019, point2gird
net_features_2019 = [
    f for f in net_2019["features"] if f["geometry"]["type"] == "LineString"
]
net_features_2024 = [
    f for f in net_2024["features"] if f["geometry"]["type"] == "LineString"
]
point2gird = {}
all_2019_lines = []
for feature in net_features_2019:
    coords = np.array(feature["geometry"]["coordinates"], dtype=np.float64)
    xy_coords = np.stack(projector(*coords.T[:2]), axis=1)  # (N, 2)
    all_2019_lines.append(LineString(xy_coords))
all_2024_lines = []
for feature in net_features_2024:
    coords = np.array(feature["geometry"]["coordinates"], dtype=np.float64)
    xy_coords = np.stack(projector(*coords.T[:2]), axis=1)  # (N, 2)
    all_2024_lines.append(LineString(xy_coords))
mid_points_2019 = [line.interpolate(0.5, normalized=True) for line in all_2019_lines]
mid_points_2024 = [line.interpolate(0.5, normalized=True) for line in all_2024_lines]
all_xys = np.array([[p.x, p.y] for p in mid_points_2019 + mid_points_2024])
min_x, min_y = np.min(all_xys, axis=0)
max_x, max_y = np.max(all_xys, axis=0)
nx, ny = ceil((max_x - min_x) / GRID_LEN), ceil((max_y - min_y) / GRID_LEN)
step_x, step_y = (max_x - min_x + 1) / nx, (max_y - min_y + 1) / ny
point2gird = {
    p: (int((p.x - min_x) / step_x), int((p.y - min_y) / step_y))
    for p in mid_points_2019 + mid_points_2024
}
all_args = [dd for dd in zip(net_features_2024, all_2024_lines, mid_points_2024)]
res = process_map(match_unit, all_args, chunksize=1000, max_workers=cpu_count())
diff_features = [r for r in res if r]
logging.info(f"# of {CITY} different roads: {len(diff_features)}")
all_build_wids = [f["properties"]["id"] for f in diff_features]
diff_features = FeatureCollection(diff_features)
pickle.dump(all_build_wids, open(ALL_CANDIDATE_WAY_PATH, "wb"))
