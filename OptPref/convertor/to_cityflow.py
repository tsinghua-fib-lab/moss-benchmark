import json
import os

import numpy as np
from pycityproto.city.map.v2.light_pb2 import LightState
from pycityproto.city.map.v2.map_pb2 import Lane
from tqdm import tqdm
from _utils.cityflow_route import AStarPath

from mosstool.type import LaneTurn, LaneType, Map, Persons

JUNC_UID_START = 3_0000_0000


def convert(
    map_file: str,
    agent_file: str,
    output_map: str,
    output_agent: str,
    time_offset: float = 0,
    use_original_route:bool=True,
):
    with open(map_file, "rb") as f:
        M = Map()
        M.ParseFromString(f.read())
    with open(agent_file, "rb") as f:
        A = Persons()
        A.ParseFromString(f.read())
    intersections = []
    roads = []
    roadnet = {"intersections": intersections, "roads": roads}
    turn_map = {
        LaneTurn.LANE_TURN_STRAIGHT: "go_straight",
        LaneTurn.LANE_TURN_LEFT: "turn_left",
        LaneTurn.LANE_TURN_RIGHT: "turn_right",
    }
    lanes_dict = {l.id: l for l in M.lanes}
    roads_dict = {r.id: r for r in M.roads}
    lane_id2road_lane_index = {}
    for r in M.roads:
        for l_index, lid in enumerate(r.lane_ids):
            l = lanes_dict[lid]
            if l.type == LaneType.LANE_TYPE_DRIVING:
                lane_id2road_lane_index[lid] = l_index

    def lane2geom(l: Lane):
        return np.array(
            [
                [
                    n.x,
                    n.y,
                ]
                for n in l.center_line.nodes
            ]
        )

    def lane2points(l: Lane):
        return [
            {
                "x": n.x,
                "y": n.y,
            }
            for n in l.center_line.nodes
        ]

    _inter_dict = {}
    _roadPointDict = {}
    for rid, r in roads_dict.items():
        r_line_nodes = lanes_dict[r.lane_ids[0]].center_line.nodes
        _roadPointDict[rid] = (r_line_nodes[0].x, r_line_nodes[0].y)
    _roadLinksDict = {}
    for j in M.junctions:
        if len(j.lane_ids) == 0:
            continue
        xy = np.vstack([lane2geom(lanes_dict[lid]) for lid in j.lane_ids])
        x, y = xy.mean(0).tolist()
        tl = []
        inter = {
            "id": str(j.id),
            "point": {"x": int(x), "y": int(y)},
            "width": 0,  # cityflow会根据width缩短道路长度
            "roads": [],
            "roadLinks": [],
            "trafficLight": {"lightphases": tl},
            "virtual": False,
        }
        _inter_dict[inter["id"]] = inter
        intersections.append(inter)
        rs = []
        for lid in j.lane_ids:
            l = lanes_dict[lid]
            for llid in l.predecessors:
                ll = lanes_dict[llid.id]
                rs.append(ll.parent_id)
            for llid in l.successors:
                ll = lanes_dict[llid.id]
                rs.append(ll.parent_id)
        rs = [r for r in rs if r < JUNC_UID_START]
        inter["roads"] = [str(i) for i in sorted(set(rs))]
        ls = [j for i in j.driving_lane_groups for j in i.lane_ids]
        assert (
            len(ls)
            == len(set(ls))
            == sum(
                lanes_dict[lid].type == LaneType.LANE_TYPE_DRIVING for lid in j.lane_ids
            )
        )
        g2l = {}
        l2g = {}
        for i, g in enumerate(j.driving_lane_groups):
            g2l[i] = list(g.lane_ids)
            for l in g.lane_ids:
                l2g[l] = i
            lane_links = []
            _lengths = []
            for i in g.lane_ids:
                l = lanes_dict[i]
                assert len(l.predecessors) == 1 and len(l.successors) == 1
                assert l.type == LaneType.LANE_TYPE_DRIVING
                p = l.predecessors[0]
                p = lane_id2road_lane_index[p.id]
                s = l.successors[0]
                s = lane_id2road_lane_index[s.id]
                _lengths.append(l.length)
                lane_links.append(
                    {
                        "startLaneIndex": p,
                        "endLaneIndex": s,
                        "points": lane2points(l) if l.length > 1 else [],
                    }
                )
            ls = []
            assert len(lane_links) > 0
            _road_weight = np.mean(_lengths) + np.mean(
                [lanes_dict[lid].length for lid in roads_dict[g.in_road_id].lane_ids]
            )  # 当前road开头到下一段road开头的权重
            _roadLinksDict[(g.in_road_id, g.out_road_id)] = (lane_links, _road_weight)
            inter["roadLinks"].append(
                {
                    "type": turn_map[g.turn],
                    "startRoad": str(g.in_road_id),
                    "endRoad": str(g.out_road_id),
                    "laneLinks": lane_links,
                }
            )
        if len(j.fixed_program.phases) > 0:
            t = j.fixed_program
            for p in t.phases:
                state = [-1] * len(g2l)
                assert len(p.states) == len(j.lane_ids)
                for s, lid in zip(p.states, j.lane_ids):
                    l = lanes_dict[lid]
                    if l.type != LaneType.LANE_TYPE_DRIVING:
                        continue
                    s = int(s == LightState.LIGHT_STATE_GREEN)
                    g = l2g[l.id]
                    if state[g] == -1:
                        state[g] = s
                    # assert state[g] == -1 or state[g] == s
                    # state[g] = s
                tl.append(
                    {
                        "time": p.duration,
                        "availableRoadLinks": [
                            i for i, j in enumerate(state) if j == 1
                        ],
                    }
                )
        else:
            tl.append(
                {
                    "time": 30,
                    "availableRoadLinks": list(range(len(j.driving_lane_groups))),
                }
            )
    _r_dict = {}
    for r in M.roads:
        start = None
        end = None
        for lid in r.lane_ids:
            l = lanes_dict[lid]
            for ll in l.predecessors:
                start = lanes_dict[ll.id].parent_id
                break
            for ll in l.successors:
                end = lanes_dict[ll.id].parent_id
                break
        if start is None:
            start = f"Start_of_{r.id}"
            x, y = lane2geom(lanes_dict[r.lane_ids[0]])[0]
            inter = {
                "id": start,
                "point": {"x": int(x), "y": int(y)},
                "width": 0,
                "roads": [],
                "roadLinks": [],
                "trafficLight": {"lightphases": []},
                "virtual": True,
            }
            _inter_dict[inter["id"]] = inter
            intersections.append(inter)
        else:
            start = str(start)
        if end is None:
            end = f"End_of_{r.id}"
            x, y = lane2geom(lanes_dict[r.lane_ids[0]])[0]
            inter = {
                "id": end,
                "point": {"x": int(x), "y": int(y)},
                "width": 0,
                "roads": [],
                "roadLinks": [],
                "trafficLight": {"lightphases": []},
                "virtual": True,
            }
            _inter_dict[inter["id"]] = inter
            intersections.append(inter)
        else:
            end = str(end)
        cur_r = {
            "id": str(r.id),
            "startIntersection": start,
            "endIntersection": end,
            "points": lane2points(lanes_dict[r.lane_ids[0]]),
            "lanes": [
                {
                    "width": lanes_dict[lid].width,
                    "maxSpeed": lanes_dict[lid].max_speed,
                }
                for lid in r.lane_ids
                if lanes_dict[lid].type == LaneType.LANE_TYPE_DRIVING
            ],
        }
        _r_dict[cur_r["id"]] = cur_r
        roads.append(cur_r)

    def convert_route(road_ids):
        def _remove_loop(route):
            next = {}
            for i, j in zip(route, route[1:]):
                next[i] = j
            next[route[-1]] = None
            r = route[:1]
            while True:
                n = next[r[-1]]
                if n is None:
                    break
                r.append(n)
            return r

        no_loop_rids = _remove_loop(road_ids)
        has_link_rids = []
        for cur_rid, next_rid in zip(no_loop_rids[:-1], no_loop_rids[1:]):
            lane_links = _roadLinksDict.get((cur_rid, next_rid), ([],))[0]
            if lane_links:
                has_link_rids = has_link_rids[:-1]
                has_link_rids.extend([cur_rid, next_rid])
            else:
                break
        return [str(i) for i in has_link_rids]

    astarPath = AStarPath(_roadLinksDict, _roadPointDict)

    def cal_route(road_ids):
        s_rid, e_rid = road_ids[0], road_ids[-1]
        route_rids = astarPath.astar_search(s_rid, e_rid)
        return [str(item) for item in route_rids]

    flow = []
    if use_original_route:
        print(f"Using original `road_ids` for agents!")
    else:
        print(f"Using new routes for agents!")
    for a in tqdm(list(A.persons)):
        _road_ids = a.schedules[0].trips[0].routes[0].driving.road_ids
        if use_original_route:
            route = convert_route(_road_ids)
        else:
            route = cal_route(_road_ids)
        if not route:
            continue
        agent = {
            "vehicle": {
                "length": a.attribute.length,
                "width": a.attribute.width,
                "maxPosAcc": a.attribute.max_acceleration,
                "maxNegAcc": -a.attribute.max_braking_acceleration,
                "usualPosAcc": a.attribute.usual_acceleration,
                "usualNegAcc": -a.attribute.usual_braking_acceleration,
                "minGap": a.vehicle_attribute.min_gap,
                "maxSpeed": a.attribute.max_speed,
                "headwayTime": 1.5,
            },
            "route": route,
            "interval": 1,
            "startTime": int(a.schedules[0].departure_time) - time_offset,
            "endTime": int(a.schedules[0].departure_time) - time_offset,
        }
        flow.append(agent)
    print(f"agent: {len(flow)}")
    os.makedirs(os.path.dirname(output_map), exist_ok=True)
    json.dump(
        roadnet,
        open(output_map, "w"),
        indent=2,
    )
    json.dump(
        flow,
        open(output_agent, "w"),
        indent=2,
    )


def remove_loop(route):
    next = {}
    for i, j in zip(route, route[1:]):
        next[i] = j
    next[route[-1]] = None
    r = route[:1]
    while True:
        n = next[r[-1]]
        if n is None:
            break
        r.append(n)
    return r


def main():
    city = "newyork"
    city2country = {
        "beijing": "china",
        "shanghai": "china",
        "newyork": "us",
        "paris": "france",
    }
    convert(
        f"./moss.map_{city2country[city]}_{city}.pb",
        f"./normal_{city}_trip.pb",
        f"./{city}_roadnet.json",
        f"./{city}_flow.json",
        time_offset=7 * 3600,
    )


if __name__ == "__main__":
    main()
