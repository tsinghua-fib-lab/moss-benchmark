from moss import Engine, LaneChange, TlPolicy, Verbosity
from moss.map import LaneTurn, LaneType, LightState
from typing import Dict, List, Tuple
from numpy.typing import NDArray
import numpy as np

__all__ = ["get_moss_engine"]

def get_moss_engine(map_file, agent_file, start_step):
    eng = Engine(
        map_file=map_file,
        agent_file=agent_file,
        start_step=start_step,
        verbose_level=Verbosity.NO_OUTPUT,
        lane_change=LaneChange.MOBIL,
        lane_veh_add_buffer_size=1400,
        lane_veh_remove_buffer_size=1000,
    )
    eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.MANUAL)
    return MossApiEngine(eng)
class MossApiEngine:
    def __init__(self, moss_engine,):
        self.moss_engine = moss_engine

    def get_current_time(
        self,
    ) -> float:  # type:ignore
        return self.moss_engine.get_current_time()
    
    def get_departed_vehicle_average_traveling_time(
        self,
    ) -> float:  # type:ignore
        return self.moss_engine.get_departed_vehicle_average_traveling_time()
    
    def get_finished_vehicle_average_traveling_time(
        self,
    ) -> float:  # type:ignore
        return self.moss_engine.get_finished_vehicle_average_traveling_time()
    
    def get_finished_vehicle_count(
        self,
    ) -> int:  # type:ignore
        return self.moss_engine.get_finished_vehicle_count()
    
    def get_junction_inout_lanes(
        self,
    ) -> Tuple[List[List[int]], List[List[int]]]:  # type:ignore
        return self.moss_engine.get_junction_inout_lanes()
    
    
    def get_junction_phase_counts(
        self,
    ) -> NDArray[np.int32]:  # type:ignore
        return self.moss_engine.get_junction_phase_counts()
    
    def get_junction_phase_lanes(
        self,
    ) -> List[List[Tuple[List[int], List[int]]]]:  # type:ignore
        return self.moss_engine.get_junction_phase_lanes()
    
    def get_lane_lengths(
        self,
    ) -> NDArray[np.int8]:  # type:ignore
        # ATTENTION:CityFlow中所有的get_lane_xxx() API只返回road lane
        return self.moss_engine.get_lane_lengths()
    
    def get_lane_vehicle_counts(
        self,
    ) -> NDArray:  # type:ignore
        return np.array(self.moss_engine.get_lane_vehicle_counts(), dtype=int)

    def get_lane_waiting_at_end_vehicle_counts(
        self, speed_threshold: float = 0.1, distance_to_end: float = 100
    ) -> NDArray:  # type:ignore
        return np.array(
            self.moss_engine.get_lane_waiting_at_end_vehicle_counts(
                speed_threshold, distance_to_end
            ),
            dtype=int,
        )

    def get_lane_waiting_vehicle_counts(
        self, speed_threshold: float = 0.1
    ) -> NDArray:  # type:ignore
        return np.array(
            self.moss_engine.get_lane_waiting_vehicle_counts(speed_threshold),
            dtype=int,
        )
    
    def set_tl_phase_batch(
        self, junction_indices: List[int], phase_indices: List[int]
    ):  # type:ignore
        self.moss_engine.set_tl_phase_batch(junction_indices, phase_indices)

    def next_step(self, n: int = 1):
        self.moss_engine.next_step(n)

    def colight_lanes_collect(self,jids):        
            M = self.moss_engine.get_map()
            js = [M.junctions[i].id for i in jids]    
            in_lanes_list = []
            out_lanes_list = []
            phase_lanes_list = []
            phase_label_list = []
            for jid in js:
                junction = M.junction_map[jid]
                phases_lane = []
                in_lane, out_lane = [], []
                labels = []
                if junction.tl:
                    tl = junction.tl
                    for phase in tl.phases:
                        lanes = [i for i, j in zip(junction.lanes, phase.states) if j == LightState.GREEN and i.type == LaneType.DRIVING and i.turn != LaneTurn.RIGHT and i.turn != LaneTurn.AROUND]
                        in_lanes = [m.predecessors[0].id for m in lanes]
                        out_lanes = [m.successors[0].id for m in lanes]
                        phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                        in_lane += in_lanes
                        out_lane += out_lanes
                        labels.append([
                            any(i.turn == LaneTurn.STRAIGHT for i in lanes),
                            any(i.turn == LaneTurn.LEFT for i in lanes)
                        ])
                in_lanes_list.append(list(set(in_lane)))
                out_lanes_list.append(list(set(out_lane)))
                phase_lanes_list.append(phases_lane)
                phase_label_list.append(labels)
            return in_lanes_list, out_lanes_list, phase_lanes_list, phase_label_list

    def get_lane_length_dict(self) -> Dict[int, float]:  # type:ignore
        M = self.moss_engine.get_map()
        return {l_id:l.length for l_id,l in enumerate(M.lanes)}
    
    def make_checkpoint(self,)->int:
        return self.moss_engine.make_checkpoint()
    
    def reset(self,cid)->int:
        return self.moss_engine.restore_checkpoint(cid)