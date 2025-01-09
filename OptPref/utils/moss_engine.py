import math
from collections import defaultdict

import numpy as np
import pycityproto.city.map.v2.light_pb2 as lightv2
import pycityproto.city.map.v2.map_pb2 as mapv2
from moss import Engine, TlPolicy, Verbosity
from mosstool.type import LaneTurn, LaneType, Map
from numpy.typing import NDArray
from numba import njit

from .decorators import timing_decorator
@njit
def _populate_lookup_array(cnt_items:list[tuple[int,int]], lane_lookup_array:np.ndarray):
    for lid, v in cnt_items:
        lane_lookup_array[lid] = v
    return lane_lookup_array
@njit
def _populate_lane_counting_dict(has_vehicle_lane_ids:np.ndarray, lane_ids:np.ndarray,):
    lane_counting_dict:dict[int,int] = {lid: 0 for lid in lane_ids}
    for lid in has_vehicle_lane_ids:
        lane_counting_dict[lid] += 1
    return np.array([lane_counting_dict[lid] for lid in lane_ids], dtype=np.int32)

__all__ = ["get_moss_engine"]


def get_moss_engine(
    map_file: str,
    person_file: str,
    start_step: int,
    name: str = "",
):
    eng = Engine(
        name=name,
        map_file=map_file,
        person_file=person_file,
        start_step=start_step,
        verbose_level=Verbosity.NO_OUTPUT,
    )
    eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.MANUAL)
    return MossApiEngine(eng)


class MossApiEngine:
    def __init__(
        self,
        moss_engine: Engine,
    ):
        self.moss_engine: Engine = moss_engine
        self.map_pb: Map = moss_engine.get_map(dict_return=False)  # type:ignore
        # lane look up
        self.lane_ids = np.array([l.id for l in self.map_pb.lanes], dtype=int)

    def get_current_time(
        self,
    ) -> float:
        return self.moss_engine.get_current_time()

    def get_departed_person_average_traveling_time(
        self,
    ) -> float:
        return self.moss_engine.get_departed_person_average_traveling_time()

    def get_finished_person_average_traveling_time(
        self,
    ) -> float:
        return self.moss_engine.get_finished_person_average_traveling_time()

    def get_finished_person_count(
        self,
    ) -> int:
        return self.moss_engine.get_finished_person_count()

    def get_junction_inout_lanes(
        self,
    ) -> tuple[list[list[int]], list[list[int]]]:
        in_lane_indexes: list[list[int]] = []
        out_lane_indexes: list[list[int]] = []
        lane_id_2_index = {l.id: i for i, l in enumerate(self.map_pb.lanes)}
        lanes_dict = {l.id: l for i, l in enumerate(self.map_pb.lanes)}
        for j in self.map_pb.junctions:
            lane_in, lane_out = [], []
            for lid in j.lane_ids:
                l = lanes_dict[lid]
                if l.type == LaneType.LANE_TYPE_DRIVING:
                    lane_in.append(lane_id_2_index[l.predecessors[0].id])
                    lane_out.append(lane_id_2_index[l.successors[0].id])
            in_lane_indexes.append(lane_in)
            out_lane_indexes.append(lane_out)
        return (in_lane_indexes, out_lane_indexes)

    def get_junction_phase_counts(
        self,
    ) -> NDArray[np.int32]:
        return self.moss_engine.get_junction_phase_counts()

    def get_junction_phase_lanes(
        self,
    ) -> list[list[tuple[list[int], list[int]]]]:
        return self.moss_engine.get_junction_phase_lanes()

    def get_lane_lengths(
        self,
    ) -> NDArray[np.float32]:
        return np.array([l.length for l in self.map_pb.lanes], dtype=np.float32)

    @timing_decorator
    def get_lane_vehicle_counts(
        self,
    ) -> NDArray:
        fetched_persons = self.moss_engine.fetch_persons()
        has_vehicle_lane_ids = fetched_persons["lane_id"]
        if len(has_vehicle_lane_ids)>0:
            lane_counts_array = _populate_lane_counting_dict(has_vehicle_lane_ids,self.lane_ids)
        else:
            lane_counts_array = np.zeros(len(self.map_pb.lanes), dtype=int)
        return lane_counts_array

    @timing_decorator
    def get_lane_waiting_at_end_vehicle_counts(
        self, speed_threshold: float = 0.1, distance_to_end: float = 100
    ) -> NDArray:
        cnt_dict = self.moss_engine.get_lane_waiting_at_end_vehicle_counts(
            speed_threshold, distance_to_end
        )
        lane_lookup_array = np.zeros(len(self.map_pb.lanes) + 1, dtype=int)
        if len(cnt_dict)>0:
            lane_lookup_array = _populate_lookup_array([(k,v) for k,v in cnt_dict.items()],lane_lookup_array)
        return lane_lookup_array[self.lane_ids]

    @timing_decorator
    def get_lane_waiting_vehicle_counts(self, speed_threshold: float = 0.1) -> NDArray:
        cnt_dict = self.moss_engine.get_lane_waiting_vehicle_counts(speed_threshold)
        lane_lookup_array = np.zeros(len(self.map_pb.lanes) + 1, dtype=int)
        if len(cnt_dict)>0:
            lane_lookup_array = _populate_lookup_array([(k,v) for k,v in cnt_dict.items()],lane_lookup_array)
        return lane_lookup_array[self.lane_ids]

    @timing_decorator
    def set_tl_phase_batch(self, junction_indices: list[int], phase_indices: list[int]):
        self.moss_engine.set_tl_phase_batch(junction_indices, phase_indices)

    @timing_decorator
    def next_step(self, n: int = 1):
        self.moss_engine.next_step(n)

    def colight_lanes_collect(
        self, jids: list[int]
    ) -> tuple[list[list[int]], list[list[int]], list[list[list]], list[list[list]]]:
        js = [self.map_pb.junctions[i].id for i in jids]
        _lanes_dict: dict[int, mapv2.Lane] = {i.id: i for i in self.map_pb.lanes}
        _juncs_dict: dict[int, mapv2.Junction] = {
            i.id: i for i in self.map_pb.junctions
        }
        in_lanes_list = []
        out_lanes_list = []
        phase_lanes_list = []
        phase_label_list = []
        for jid in js:
            junction: mapv2.Junction = _juncs_dict[jid]
            phases_lane = []
            in_lane, out_lane = [], []
            labels = []
            if junction.fixed_program:
                tl: lightv2.TrafficLight = junction.fixed_program
                for phase in tl.phases:
                    lanes = [
                        _lanes_dict[lid]
                        for lid, j in zip(junction.lane_ids, phase.states)
                        if j == lightv2.LIGHT_STATE_GREEN
                        and _lanes_dict[lid].type == LaneType.LANE_TYPE_DRIVING
                        and _lanes_dict[lid].turn != LaneTurn.LANE_TURN_RIGHT
                        and _lanes_dict[lid].turn != LaneTurn.LANE_TURN_AROUND
                    ]
                    in_lanes = [m.predecessors[0].id for m in lanes]
                    out_lanes = [m.successors[0].id for m in lanes]
                    phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                    in_lane += in_lanes
                    out_lane += out_lanes
                    labels.append(
                        [
                            any(i.turn == LaneTurn.LANE_TURN_STRAIGHT for i in lanes),
                            any(i.turn == LaneTurn.LANE_TURN_LEFT for i in lanes),
                        ]
                    )
            in_lanes_list.append(list(set(in_lane)))
            out_lanes_list.append(list(set(out_lane)))
            phase_lanes_list.append(phases_lane)
            phase_label_list.append(labels)
        return in_lanes_list, out_lanes_list, phase_lanes_list, phase_label_list

    def get_lane_length_dict(self) -> dict[int, float]:  # type:ignore
        M: Map = self.moss_engine.get_map(dict_return=False)  # type: ignore
        return {l_id: l.length for l_id, l in enumerate(M.lanes)}

    def make_checkpoint(
        self,
    ) -> int:
        return self.moss_engine.make_checkpoint()

    def reset(self, cid: int):
        return self.moss_engine.restore_checkpoint(cid)

    def advanced_mplight_frap_lanes_collect(self, js: list[int]) -> tuple[
        list[list[int]],
        list[list[int]],
        list[list[list]],
        list[list[list]],
        list[list[list]],
        list[list[list]],
    ]:
        in_lanes_list = []
        out_lanes_list = []
        phase_lanes_list = []
        phase_lanes_A_list, phase_lanes_B_list = [], []
        phase_label_list = []
        _lanes_dict: dict[int, mapv2.Lane] = {i.id: i for i in self.map_pb.lanes}
        _juncs_dict: dict[int, mapv2.Junction] = {
            i.id: i for i in self.map_pb.junctions
        }

        def _lane_start_angle(l: mapv2.Lane) -> float:
            _nodes = l.center_line.nodes
            n0, n1 = _nodes[0], _nodes[1]
            return math.atan2((n1.y - n0.y), (n1.x - n0.x))

        for jid in js:
            junction = _juncs_dict[jid]
            phases_lane = []
            phases_lanes_A, phases_lanes_B = [], []
            in_lane, out_lane = [], []
            in_lane_A, in_lane_B, out_lane_A, out_lane_B = [], [], [], []
            labels = []
            if junction.fixed_program:
                tl: lightv2.TrafficLight = junction.fixed_program
                for phase in tl.phases:
                    lanes = [
                        _lanes_dict[lid]
                        for lid, j in zip(junction.lane_ids, phase.states)
                        if j == lightv2.LIGHT_STATE_GREEN
                        and _lanes_dict[lid].type == LaneType.LANE_TYPE_DRIVING
                        and _lanes_dict[lid].turn != LaneTurn.LANE_TURN_RIGHT
                        and _lanes_dict[lid].turn != LaneTurn.LANE_TURN_AROUND
                    ]
                    in_lanes = [m.predecessors[0].id for m in lanes]
                    out_lanes = [m.successors[0].id for m in lanes]
                    phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                    in_lane += in_lanes
                    out_lane += out_lanes
                    labels.append(
                        [
                            any(i.turn == LaneTurn.LANE_TURN_STRAIGHT for i in lanes),
                            any(i.turn == LaneTurn.LANE_TURN_LEFT for i in lanes),
                        ]
                    )
                    # 对lanes根据具体的travel movement进行分类
                    in_angles = [_lane_start_angle(lane) for lane in lanes]
                    in_angles = np.array(in_angles) - min(in_angles)
                    indexes_tmB = (in_angles >= np.pi / 2) & (
                        in_angles <= 3 * np.pi / 2
                    )
                    indexes_tmA = [not i for i in indexes_tmB]
                    lanes_tmA, lanes_tmB = (
                        np.array(lanes)[indexes_tmA],
                        np.array(lanes)[indexes_tmB],
                    )
                    in_lane_A = [m.predecessors[0].id for m in lanes_tmA]
                    in_lane_B = [m.predecessors[0].id for m in lanes_tmB]
                    out_lane_A = [m.successors[0].id for m in lanes_tmA]
                    out_lane_B = [m.successors[0].id for m in lanes_tmB]
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

    @property
    def lane_count(
        self,
    ) -> int:
        return self.moss_engine.lane_count

    @property
    def person_count(
        self,
    ) -> int:
        return self.moss_engine.person_count

    @property
    def road_count(
        self,
    ) -> int:
        return self.moss_engine.road_count

    @property
    def junction_count(
        self,
    ) -> int:
        return self.moss_engine.junction_count
