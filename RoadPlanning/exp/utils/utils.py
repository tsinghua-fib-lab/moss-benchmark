import asyncio
import logging
import random
from copy import deepcopy
from typing import Optional, cast

import geojson
import numpy as np
import pyproj
from geojson import Feature, FeatureCollection, LineString, MultiPoint
from mosstool.trip.route import RoutingClient
from mosstool.type import Map, Person, Persons
from mosstool.util.format_converter import dict2pb, pb2dict
from pycityproto.city.geo.v2.geo_pb2 import LanePosition, Position
from pycityproto.city.person.v1.person_pb2 import Person
from pycityproto.city.routing.v2.routing_pb2 import RouteType
from pycityproto.city.routing.v2.routing_service_pb2 import GetRouteRequest
from pycityproto.city.trip.v2.trip_pb2 import Schedule, TripMode
from shapely.geometry import LineString as sLineString
from shapely.strtree import STRtree
from tqdm import tqdm

LANE_TYPE_DRIVE = 1
LANE_TYPE_WALK = 2
_TYPE_MAP = {
    TripMode.TRIP_MODE_DRIVE_ONLY: RouteType.ROUTE_TYPE_DRIVING,
    TripMode.TRIP_MODE_BIKE_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_BUS_WALK: RouteType.ROUTE_TYPE_WALKING,
    TripMode.TRIP_MODE_WALK_ONLY: RouteType.ROUTE_TYPE_WALKING,
}


async def my_pre_route(
    client: RoutingClient, person: Person, sub_eta: bool, in_place: bool = False
) -> Person:
    if not in_place:
        p = Person()
        p.CopyFrom(person)
        person = p
    start = person.home
    departure_time = None
    all_schedules = list(person.schedules)
    person.ClearField("schedules")
    for schedule in all_schedules:
        schedule = cast(Schedule, schedule)
        if schedule.HasField("departure_time"):
            departure_time = schedule.departure_time
        if schedule.loop_count != 1:
            logging.warning(
                "Schedule is not a one-time trip, departure time is not accurate, no pre-calculation is performed"
            )
            start = schedule.trips[-1].end
            continue
        good_trips = []
        for trip in schedule.trips:
            last_departure_time = departure_time
            # Cover departure time
            if trip.HasField("departure_time"):
                departure_time = trip.departure_time
            if departure_time is None:
                continue
            if not trip.mode == TripMode.TRIP_MODE_DRIVE_ONLY:
                departure_time = last_departure_time
                continue
            # build request
            res = await client.GetRoute(
                GetRouteRequest(
                    type=_TYPE_MAP[trip.mode],
                    start=start,
                    end=trip.end,
                    time=departure_time,
                )
            )
            if len(res.journeys) == 0:
                # logging.warning("No route found")
                departure_time = last_departure_time
            else:
                # append directly
                good_trips.append(trip)
                trip.ClearField("routes")
                trip.routes.MergeFrom(res.journeys)
                # update start position
                start = trip.end
                # Set departure time invalid
                departure_time = None
        if len(good_trips) > 0:
            good_trips = good_trips[:1]
            good_schedule = cast(Schedule, person.schedules.add())
            good_schedule.CopyFrom(schedule)
            good_schedule.ClearField("trips")
            if sub_eta:
                good_schedule.departure_time = (
                    good_schedule.departure_time - good_trips[0].routes[0].driving.eta
                )
            good_schedule.trips.extend(good_trips)
            break
    return person


def _road_id2lane_pos(
    road_id0, road_id1, rng, m_lanes, m_roads, m_lane_lengths: Optional[dict] = None
):
    if not road_id0 == road_id1:
        drive_lane_ids_0 = [
            lid
            for lid in m_roads[road_id0]["lane_ids"]
            if m_lanes[lid]["type"] == LANE_TYPE_DRIVE
        ]
        select_lid_0 = rng.choice(drive_lane_ids_0)
        if m_lane_lengths is not None:
            select_len_0 = m_lane_lengths[select_lid_0]
        else:
            select_len_0 = m_lanes[select_lid_0]["length"]
        select_s_0 = rng.uniform(0.1, 0.3) * select_len_0
        drive_lane_ids_1 = [
            lid
            for lid in m_roads[road_id1]["lane_ids"]
            if m_lanes[lid]["type"] == LANE_TYPE_DRIVE
        ]
        select_lid_1 = rng.choice(drive_lane_ids_1)
        if m_lane_lengths is not None:
            select_len_1 = m_lane_lengths[select_lid_1]
        else:
            select_len_1 = m_lanes[select_lid_1]["length"]
        select_s_1 = rng.uniform(0.6, 0.9) * select_len_1
    else:
        drive_lane_ids_0 = [
            lid
            for lid in m_roads[road_id0]["lane_ids"]
            if m_lanes[lid]["type"] == LANE_TYPE_DRIVE
        ]
        select_lid_0 = rng.choice(drive_lane_ids_0)
        if m_lane_lengths is not None:
            select_len_0 = m_lane_lengths[select_lid_0]
        else:
            select_len_0 = m_lanes[select_lid_0]["length"]
        select_lid_1 = select_lid_0
        if m_lane_lengths is not None:
            select_len_1 = m_lane_lengths[select_lid_1]
        else:
            select_len_1 = m_lanes[select_lid_1]["length"]
        select_s_0 = rng.uniform(0.1, 0.3) * select_len_0
        select_s_1 = rng.uniform(0.6, 0.9) * select_len_1
    if select_lid_0 == select_lid_1 and select_s_0 > select_s_1:
        select_s_0, select_s_1 = select_s_1, select_s_0
    return (select_lid_0, select_s_0, select_lid_1, select_s_1)


async def fetch_trip_route(
    persons,
    m_dict,
    listen: str,
    output_path_home: str,
    output_path_work: str,
    max_num: int,
    other_persons,
    transfer_to_lane_pos: bool = True,
):
    client = RoutingClient(listen)
    all_persons = []
    m_lanes = {d["id"]: d for d in m_dict["lanes"]}
    m_roads = {d["id"]: d for d in m_dict["roads"]}
    random.seed(0)
    args = []
    args += list(
        random.sample(
            [(p, False) for p in persons], min(max_num // 2 + 1_0000, len(persons))
        )
    )
    args += list(
        random.sample(
            [(p, True) for p in other_persons],
            min(max_num // 2 + 1_0000, len(other_persons)),
        )
    )
    BATCH = 15000
    for i in tqdm(range(0, len(args), BATCH)):
        ps = await asyncio.gather(
            *[my_pre_route(client, p, sub_eta) for (p, sub_eta) in args[i: i + BATCH]]
        )
        all_persons.extend(ps)
    ok_persons = []
    for p in all_persons:
        if len(p.schedules) == 0:
            continue
        if len(p.schedules[0].trips) == 0:
            continue
        BAD_PERSON = False
        start_id = p.home.lane_position.lane_id
        end_id = p.schedules[0].trips[0].end.lane_position.lane_id
        trip_mode = p.schedules[0].trips[0].mode
        if trip_mode in [
            TripMode.TRIP_MODE_BIKE_WALK,
            TripMode.TRIP_MODE_BUS_WALK,
            TripMode.TRIP_MODE_WALK_ONLY,
        ]:
            if (
                not m_lanes[start_id]["type"] == LANE_TYPE_WALK
                or not m_lanes[end_id]["type"] == LANE_TYPE_WALK
            ):
                BAD_PERSON = True
        else:
            if (
                not m_lanes[start_id]["type"] == LANE_TYPE_DRIVE
                or not m_lanes[end_id]["type"] == LANE_TYPE_DRIVE
            ):
                BAD_PERSON = True
        if BAD_PERSON:
            continue
        ok_persons.append(p)
    ok_persons = ok_persons[:max_num]
    pb = Persons(persons=ok_persons)
    new_to_home_persons = []
    to_home_p_id = 0
    new_to_work_persons = []
    to_work_p_id = 0
    for p in tqdm(pb.persons):
        try:
            if transfer_to_lane_pos:
                all_road_ids = p.schedules[0].trips[0].routes[0].driving.road_ids
                rng = np.random.default_rng(p.id)
                road_id_0 = all_road_ids[0]
                road_id_1 = all_road_ids[-1]
                (select_lid_0, select_s_0, select_lid_1, select_s_1) = (
                    _road_id2lane_pos(road_id_0, road_id_1, rng, m_lanes, m_roads)
                )
                p.ClearField("home")
                p.home.CopyFrom(
                    Position(
                        lane_position=LanePosition(
                            lane_id=select_lid_0,
                            s=select_s_0,
                        )
                    )
                )
                trip = p.schedules[0].trips[0]
                trip.ClearField("end")
                trip.end.CopyFrom(
                    Position(
                        lane_position=LanePosition(
                            lane_id=select_lid_1,
                            s=select_s_1,
                        )
                    )
                )
            depart_time = p.schedules[0].departure_time
            if depart_time < 12 * 3600:
                p.id = to_work_p_id
                new_to_work_persons.append(p)
                to_work_p_id += 1
            else:
                p.id = to_home_p_id
                new_to_home_persons.append(p)
                to_home_p_id += 1
        except:
            continue
    with open(output_path_home, "wb") as f:
        to_home_pb = Persons(persons=new_to_home_persons)
        f.write(to_home_pb.SerializeToString())
    with open(output_path_work, "wb") as f:
        to_work_pb = Persons(persons=new_to_work_persons)
        f.write(to_work_pb.SerializeToString())


def topo_dict2net(new_topo_dict):
    geos = []
    for geo_type, dicts in new_topo_dict.items():
        for feature in dicts.values():
            coords = feature["geometry"]["coordinates"]
            if geo_type == "LineString":
                geo = LineString(coords)
            else:
                geo = MultiPoint(coords)
                prop = feature["properties"]
                if not prop["in_ways"] and not prop["out_ways"]:
                    continue
            geos.append(
                Feature(
                    id=feature["id"],
                    properties=feature["properties"],
                    geometry=geo,
                )
            )
    return FeatureCollection(geos)


def get_geojson_data(roadnet_path, proj_str):
    projector = pyproj.Proj(proj_str)
    with open(roadnet_path, "r") as f:
        raw_roadnet = geojson.load(f)
    raw_roadnet = FeatureCollection(raw_roadnet)
    orig_topo_dict = {k: {} for k in ("MultiPoint", "LineString")}
    for feature in raw_roadnet["features"]:
        if "id" not in feature:
            feature["id"] = feature["properties"]["id"]
        feature_type = feature["geometry"]["type"]
        orig_topo_dict[feature_type][feature["id"]] = feature
    way_id2junc_id = {
        way_id: {"in_ways": [], "out_ways": []}
        for way_id, _ in orig_topo_dict["LineString"].items()
    }
    way_id2line = {}
    way_id2idx = {}
    for idx, (way_id, feature) in enumerate(orig_topo_dict["LineString"].items()):
        coords = np.array(feature["geometry"]["coordinates"], dtype=np.float64)
        xy_coords = np.stack(projector(*coords.T[:2]), axis=1)  # (N, 2)
        way_id2line[way_id] = sLineString(xy_coords)
        way_id2idx[way_id] = idx
    for junc_id, feature in orig_topo_dict["MultiPoint"].items():
        prop = feature["properties"]
        for way_id in prop["in_ways"]:
            way_id2junc_id[way_id]["in_ways"].append(junc_id)
        for way_id in prop["out_ways"]:
            way_id2junc_id[way_id]["out_ways"].append(junc_id)
    return (orig_topo_dict, way_id2junc_id, way_id2line, way_id2idx)


def delete_way(way_id, topo_dict, way_id2junc_id):
    if way_id in topo_dict["LineString"]:
        del topo_dict["LineString"][way_id]
    in_ways_junc_ids = way_id2junc_id[way_id]["in_ways"]
    out_ways_junc_ids = way_id2junc_id[way_id]["out_ways"]
    for junc_id in in_ways_junc_ids:
        junc = topo_dict["MultiPoint"][junc_id]
        junc["properties"]["in_ways"] = [
            wid for wid in junc["properties"]["in_ways"] if not wid == way_id
        ]
    for junc_id in out_ways_junc_ids:
        junc = topo_dict["MultiPoint"][junc_id]
        junc["properties"]["out_ways"] = [
            wid for wid in junc["properties"]["out_ways"] if not wid == way_id
        ]


def way_id2road_id(way_id, topo_dict):
    road_id_offset = 2_0000_0000
    for rid, wid in enumerate(topo_dict["LineString"].keys(), start=road_id_offset):
        if wid == way_id:
            return rid
    return None


def build_new_map(
    orig_topo_dict, ORIG_MAP, opt_way_ids, rec_x_values, way_id2junc_id, CITY
):
    new_topo_dict = deepcopy(orig_topo_dict)
    all_road_ids = []
    for opt_wid, x_type in zip(opt_way_ids, rec_x_values):
        if x_type == 1:
            pass
        elif x_type == 0:
            all_road_ids.append(way_id2road_id(topo_dict=new_topo_dict, way_id=opt_wid))
    all_road_ids = [rid for rid in all_road_ids if rid]
    new_map_dict = pb2dict(ORIG_MAP)
    lanes = {d["id"]: d for d in new_map_dict["lanes"]}
    roads = {d["id"]: d for d in new_map_dict["roads"]}
    new_map_dict["aois"] = []
    for _, ll in lanes.items():
        ll["aoi_ids"] = []
    for rid in all_road_ids:
        road = roads[rid]
        for lid in road["lane_ids"]:
            lanes[lid]["length"] = 1e10
    new_map_pb = dict2pb(new_map_dict, Map())
    return (new_map_dict, new_map_pb)


def get_home_and_work_persons(orig_person_path, orig_map_dict, new_map_dict):
    orig_ratio_persons_pb = Persons()
    with open(orig_person_path, "rb") as f:
        orig_ratio_persons_pb.ParseFromString(f.read())
    orig_lanes_dict = {}
    for ll in orig_map_dict["lanes"]:
        if ll["type"] == LANE_TYPE_DRIVE:
            lid = ll["id"]
            line = sLineString([(n["x"], n["y"]) for n in ll["center_line"]["nodes"]])
            orig_lanes_dict[lid] = line
    new_tree_id2lane_id = {}
    new_lanes_dict = {}
    new_geos = []
    new_tree_id = 0
    m_lane_lengths = {}
    for ll in new_map_dict["lanes"]:
        lid = ll["id"]
        line = sLineString([(n["x"], n["y"]) for n in ll["center_line"]["nodes"]])
        m_lane_lengths[lid] = line.length
        if ll["type"] == LANE_TYPE_DRIVE and ll["parent_id"] < 3_0000_0000:
            new_lanes_dict[lid] = (line, ll["parent_id"])
            new_geos.append(line)
            new_tree_id2lane_id[new_tree_id] = lid
            new_tree_id += 1
    new_lane_tree = STRtree(new_geos)
    to_home_persons = []
    to_work_persons = []
    m_lanes = {d["id"]: d for d in new_map_dict["lanes"]}
    m_roads = {d["id"]: d for d in new_map_dict["roads"]}
    for p in tqdm(orig_ratio_persons_pb.persons):
        home_lid = p.home.lane_position.lane_id
        home_s = p.home.lane_position.s
        home_point = orig_lanes_dict[home_lid].interpolate(home_s)
        home_tids = new_lane_tree.query(home_point.buffer(200))
        trip = p.schedules[0].trips[0]
        end_lid = trip.end.lane_position.lane_id
        end_s = trip.end.lane_position.s
        end_point = orig_lanes_dict[end_lid].interpolate(end_s)
        end_tids = new_lane_tree.query(end_point.buffer(200))
        if len(home_tids) > 0 and len(end_tids) > 0:
            rng = np.random.default_rng(p.id)
            temp_home_lid = new_tree_id2lane_id[home_tids[0]]
            _, new_home_rid = new_lanes_dict[temp_home_lid]
            temp_end_lid = new_tree_id2lane_id[end_tids[0]]
            _, new_end_rid = new_lanes_dict[temp_end_lid]
            (new_home_lid, new_home_s, new_end_lid, new_end_s) = _road_id2lane_pos(
                new_home_rid, new_end_rid, rng, m_lanes, m_roads, m_lane_lengths
            )
            p.ClearField("home")
            p.home.CopyFrom(
                Position(
                    lane_position=LanePosition(
                        lane_id=new_home_lid,
                        s=new_home_s,
                    )
                )
            )
            trip.ClearField("end")
            trip.end.CopyFrom(
                Position(
                    lane_position=LanePosition(
                        lane_id=new_end_lid,
                        s=new_end_s,
                    )
                )
            )
        else:
            continue
        if p.schedules[0].departure_time < 12 * 3600:
            to_work_persons.append(p)
        else:
            to_home_persons.append(p)
    return (to_work_persons, to_home_persons, len(orig_ratio_persons_pb.persons))


def show_best_result(atts, tps, algo):
    min_index = min(enumerate(atts), key=lambda x: x[1])[0]
    if algo == "random":
        return (np.mean(atts), np.mean(tps))
    else:
        return (atts[min_index], tps[min_index])
