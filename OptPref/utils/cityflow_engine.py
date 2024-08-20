import json
import os
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from shapely.geometry import LineString

import cityflow

__all__ = ["get_cityflow_engine"]


def _get_line_angle(line: LineString):
    def _get_end_vector(line: LineString):
        """Get the end vector of a LineString"""
        return np.array(line.coords[-1]) - np.array(line.coords[-2])

    v = _get_end_vector(line)
    angle = np.arctan2(v[1], v[0])
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle


def get_cityflow_engine(
    config_file: str, thread_num: int = 64, offset_time: float = 0
):
    eng = cityflow.Engine(config_file=config_file, thread_num=thread_num)
    return CityFlowApiEngine(
        cityflow_engine=eng, offset_time=offset_time, config_file=config_file
    )


class CityFlowApiEngine:
    def __init__(self, cityflow_engine, offset_time: float, config_file: str):
        self.cityflow_engine = cityflow_engine
        self.offset_time = offset_time
        with open(config_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            roadnet_file = os.path.join(data["dir"], data["roadnetFile"])
        with open(roadnet_file, "r", encoding="utf-8") as file:
            self.roadnet = json.load(file)

    def get_current_time(
        self,
    ) -> float:  # type:ignore
        # 模拟时间＋偏移时间
        return self.cityflow_engine.get_current_time() + self.offset_time

    def get_departed_vehicle_average_traveling_time(
        self,
    ) -> float:  # type:ignore
        return self.cityflow_engine._get_departed_vehicle_average_traveling_time()

    def get_finished_vehicle_average_traveling_time(
        self,
    ) -> float:  # type:ignore
        return self.cityflow_engine._get_finished_vehicle_average_traveling_time()

    def get_finished_vehicle_count(
        self,
    ) -> int:  # type:ignore
        return self.cityflow_engine._get_finished_vehicle_count()

    def get_junction_inout_lanes(
        self,
    ) -> Tuple[List[List[int]], List[List[int]]]:  # type:ignore
        _in_lanes = []
        _out_lanes = []
        for _i, _o in self.cityflow_engine._get_junction_inout_lanes():
            _in_lanes.append(_i)
            _out_lanes.append(_o)
        return (_in_lanes, _out_lanes)

    def get_junction_phase_counts(
        self,
    ) -> NDArray[np.int32]:  # type:ignore
        return self.cityflow_engine._get_junction_phase_counts()

    def get_junction_phase_lanes(
        self,
    ) -> List[List[Tuple[List[int], List[int]]]]:  # type:ignore
        return self.cityflow_engine._get_junction_phase_lanes()

    def get_lane_lengths(
        self,
    ) -> NDArray[np.int8]:  # type:ignore
        # ATTENTION:CityFlow中所有的get_lane_xxx() API只返回road lane
        return self.cityflow_engine._get_lane_lengths()

    def get_lane_vehicle_counts(
        self,
    ) -> NDArray:  # type:ignore
        return np.array(self.cityflow_engine._get_lane_vehicle_counts(), dtype=int)

    def get_lane_waiting_at_end_vehicle_counts(
        self, speed_threshold: float = 0.1, distance_to_end: float = 100
    ) -> NDArray:  # type:ignore
        return np.array(
            self.cityflow_engine._get_lane_waiting_at_end_vehicle_counts(
                speed_threshold, distance_to_end
            ),
            dtype=int,
        )

    def get_lane_waiting_vehicle_counts(
        self, speed_threshold: float = 0.1
    ) -> NDArray:  # type:ignore
        return np.array(
            self.cityflow_engine._get_lane_waiting_vehicle_counts(speed_threshold),
            dtype=int,
        )

    def set_tl_phase_batch(
        self, junction_indices: List[int], phase_indices: List[int]
    ):  # type:ignore
        self.cityflow_engine._set_tl_phase_batch(junction_indices, phase_indices)

    def next_step(self, n: int = 1):
        self.cityflow_engine._next_step(n)

    def frap_lanes_collect(self, jids):
        # FRAP和advanced_mplight
        _jids_set = set(jids)
        _intersections = self.roadnet["intersections"]
        in_lanes_list = []
        out_lanes_list = []
        phase_lanes_list = []
        phase_label_list = []
        phase_lanes_A_list, phase_lanes_B_list = [], []
        _lane_id_dict = (
            self._road_id_and_idx_to_lane_id_dict()
        )  # (road_id,idx)->lane_id
        _lane_shapely_dict = self._lane_id_to_shapely()  # lane_id->LineString

        for idx, inter in enumerate(_intersections):
            if idx not in _jids_set:
                continue
            phases_lane = []
            phases_lanes_A, phases_lanes_B = [], []
            in_lane, out_lane = [], []
            in_lane_A, in_lane_B, out_lane_A, out_lane_B = [], [], [], []
            labels = []
            if not inter["virtual"] and len(inter["trafficLight"]["lightphases"]) > 0:
                interRoadLinks = inter["roadLinks"]
                for _phase in inter["trafficLight"]["lightphases"]:
                    availabeRoadLinks = [
                        link
                        for idx, link in enumerate(interRoadLinks)
                        if idx in _phase["availableRoadLinks"]
                        and link["type"]
                        in {
                            "go_straight",
                            "turn_left",
                        }
                    ]
                    availabeLaneLinks = [
                        (
                            r_link["startRoad"],
                            r_link["endRoad"],
                            r_link.get("direction", 0),
                            l_link,
                        )
                        for r_link in availabeRoadLinks
                        for l_link in r_link["laneLinks"]
                    ]
                    if len(availabeLaneLinks) == 0:
                        continue
                    in_lanes = [
                        _lane_id_dict[(s_r, link["startLaneIndex"])]
                        for (s_r, e_r, dir, link) in availabeLaneLinks
                    ]
                    out_lanes = [
                        _lane_id_dict[(e_r, link["endLaneIndex"])]
                        for (s_r, e_r, dir, link) in availabeLaneLinks
                    ]
                    phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                    in_lane += in_lanes
                    out_lane += out_lanes
                    labels.append(
                        [
                            any(i["type"] == "go_straight" for i in availabeRoadLinks),
                            any(i["type"] == "turn_left" for i in availabeRoadLinks),
                        ]
                    )
                    in_angles = [
                        _get_line_angle(_lane_shapely_dict[in_lid])
                        for in_lid in in_lanes
                    ]
                    in_angles = np.array(in_angles) - min(in_angles)
                    lanes_tmB = (in_angles >= np.pi / 2) & (in_angles <= 3 * np.pi / 2)
                    lanes_tmA = [not i for i in lanes_tmB]
                    lanes_tmA, lanes_tmB = (
                        np.array(availabeLaneLinks)[lanes_tmA],
                        np.array(availabeLaneLinks)[lanes_tmB],
                    )
                    in_lane_A = [
                        _lane_id_dict[(s_r, link["startLaneIndex"])]
                        for (s_r, e_r, dir, link) in lanes_tmA
                    ]
                    in_lane_B = [
                        _lane_id_dict[(s_r, link["startLaneIndex"])]
                        for (s_r, e_r, dir, link) in lanes_tmB
                    ]
                    out_lane_A = [
                        _lane_id_dict[(e_r, link["endLaneIndex"])]
                        for (s_r, e_r, dir, link) in lanes_tmA
                    ]
                    out_lane_B = [
                        _lane_id_dict[(e_r, link["endLaneIndex"])]
                        for (s_r, e_r, dir, link) in lanes_tmB
                    ]
                    phases_lanes_A.append([list(set(in_lane_A)), list(set(out_lane_A))])
                    phases_lanes_B.append([list(set(in_lane_B)), list(set(out_lane_B))])
            in_lanes_list.append(list(set(in_lane)))
            out_lanes_list.append(list(set(out_lane)))
            phase_lanes_list.append(phases_lane)
            phase_lanes_A_list.append(phases_lanes_A)
            phase_lanes_B_list.append(phases_lanes_B)
            phase_label_list.append(labels)
        return (
            in_lanes_list,
            out_lanes_list,
            phase_lanes_list,
            phase_label_list,
            phase_lanes_A_list,
            phase_lanes_B_list,
        )

    def colight_lanes_collect(self, jids):
        _jids_set = set(jids)
        _intersections = self.roadnet["intersections"]
        in_lanes_list = []
        out_lanes_list = []
        phase_lanes_list = []
        phase_label_list = []
        _lane_id_dict = (
            self._road_id_and_idx_to_lane_id_dict()
        )  # (road_id,idx)->lane_id

        for idx, inter in enumerate(_intersections):
            if idx not in _jids_set:
                continue
            phases_lane = []
            in_lane, out_lane = [], []
            labels = []
            if not inter["virtual"] and len(inter["trafficLight"]["lightphases"]) > 0:
                interRoadLinks = inter["roadLinks"]
                for _phase in inter["trafficLight"]["lightphases"]:
                    availabeRoadLinks = [
                        link
                        for idx, link in enumerate(interRoadLinks)
                        if idx in _phase["availableRoadLinks"]
                        and link["type"]
                        in {
                            "go_straight",
                            "turn_left",
                        }
                    ]
                    availabeLaneLinks = [
                        (
                            r_link["startRoad"],
                            r_link["endRoad"],
                            r_link.get("direction", 0),
                            l_link,
                        )
                        for r_link in availabeRoadLinks
                        for l_link in r_link["laneLinks"]
                    ]
                    if len(availabeLaneLinks) == 0:
                        continue
                    in_lanes = [
                        _lane_id_dict[(s_r, link["startLaneIndex"])]
                        for (s_r, e_r, dir, link) in availabeLaneLinks
                    ]
                    out_lanes = [
                        _lane_id_dict[(e_r, link["endLaneIndex"])]
                        for (s_r, e_r, dir, link) in availabeLaneLinks
                    ]
                    phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                    in_lane += in_lanes
                    out_lane += out_lanes
                    labels.append(
                        [
                            any(i["type"] == "go_straight" for i in availabeRoadLinks),
                            any(i["type"] == "turn_left" for i in availabeRoadLinks),
                        ]
                    )
            in_lanes_list.append(list(set(in_lane)))
            out_lanes_list.append(list(set(out_lane)))
            phase_lanes_list.append(phases_lane)
            phase_label_list.append(labels)
        return in_lanes_list, out_lanes_list, phase_lanes_list, phase_label_list

    def get_lane_length_dict(self) -> Dict[int, float]:  # type:ignore
        # ATTENTION: 只包含road lane
        lane_id = 0
        _length_dict = {}
        for r in self.roadnet["roads"]:
            _road_line = LineString([(p["x"], p["y"]) for p in r["points"]])
            _lane_length = _road_line.length
            for _ in r["lanes"]:
                _length_dict[lane_id] = _lane_length
                lane_id += 1
        return _length_dict

    def _road_id_and_idx_to_lane_id_dict(self):
        _lane_id_dict = {}
        lane_id = 0
        for r in self.roadnet["roads"]:
            for idx, _ in enumerate(r["lanes"]):
                _lane_id_dict[(r["id"], idx)] = lane_id
                lane_id += 1
        return _lane_id_dict

    def _lane_id_to_shapely(self):
        lane_id = 0
        _shapely_dict = {}
        for r in self.roadnet["roads"]:
            _road_line = LineString([(p["x"], p["y"]) for p in r["points"]])
            for _ in r["lanes"]:
                _shapely_dict[lane_id] = _road_line
                lane_id += 1
        return _shapely_dict
    def make_checkpoint(self,)->int:
        return 0
    def reset(self,cid)->int:
        return self.cityflow_engine.reset(cid)