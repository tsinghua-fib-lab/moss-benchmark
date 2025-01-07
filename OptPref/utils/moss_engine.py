from moss import Engine, TlPolicy, Verbosity
from mosstool.type import LaneTurn, LaneType,Map
import pycityproto.city.map.v2.light_pb2 as lightv2
from numpy.typing import NDArray
import numpy as np

__all__ = ["get_moss_engine"]

def get_moss_engine(map_file, person_file, start_step):
    eng = Engine(
        name="OptPerf",
        map_file=map_file,
        person_file=person_file,
        start_step=start_step,
        verbose_level=Verbosity.NO_OUTPUT,
    )
    eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.MANUAL)
    return MossApiEngine(eng)
class MossApiEngine:
    def __init__(self, moss_engine:Engine,):
        self.moss_engine:Engine = moss_engine
        self.map_pb:Map = moss_engine.get_map(dict_return=False) # type:ignore

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
    
    def get_finished_vehicle_count(
        self,
    ) -> int:  
        return self.moss_engine.get_finished_person_count()
    
    def get_junction_inout_lanes(
        self,
    ) -> tuple[list[list[int]], list[list[int]]]:
        in_lane_indexes:list[list[int]] = []
        out_lane_indexes:list[list[int]] = []
        lane_id_2_index = {l.id:i for i,l in enumerate(self.map_pb.lanes)}
        lanes_dict = {l.id:l for i,l in enumerate(self.map_pb.lanes)}
        for j in self.map_pb.junctions:
            lane_in, lane_out = [],[]
            for lid in j.lane_ids:
                l =  lanes_dict[lid]
                if l.type == LaneType.LANE_TYPE_DRIVING:
                    lane_in.append(lane_id_2_index[l.predecessors[0].id])
                    lane_out.append(lane_id_2_index[l.successors[0].id])      
            in_lane_indexes.append(lane_in)     
            out_lane_indexes.append(lane_out)                          
        return (in_lane_indexes,out_lane_indexes)
    
    
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
        return np.array([l.length for l in self.map_pb.lanes],dtype=np.float32)
    
    def get_lane_vehicle_counts(
        self,
    ) -> NDArray: 
        return np.array(self.moss_engine.get_lane_vehicle_counts(), dtype=int)

    def get_lane_waiting_at_end_vehicle_counts(
        self, speed_threshold: float = 0.1, distance_to_end: float = 100
    ) -> NDArray:  # type:ignore
        
        cnt_dict = self.moss_engine.get_lane_waiting_at_end_vehicle_counts(
                speed_threshold, distance_to_end
            )
        cnt = [cnt_dict[l.id] for l in self.map_pb.lanes]
        return np.array(
            cnt,
            dtype=int,
        )

    def get_lane_waiting_vehicle_counts(
        self, speed_threshold: float = 0.1
    ) -> NDArray: 
        return np.array(
            self.moss_engine.get_lane_waiting_vehicle_counts(speed_threshold),
            dtype=int,
        )
    
    def set_tl_phase_batch(
        self, junction_indices: list[int], phase_indices: list[int]
    ):
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

    def get_lane_length_dict(self) -> dict[int, float]:  # type:ignore
        M = self.moss_engine.get_map()
        return {l_id:l.length for l_id,l in enumerate(M.lanes)}
    
    def make_checkpoint(self,)->int:
        return self.moss_engine.make_checkpoint()
    
    def reset(self,cid:int):
        return self.moss_engine.restore_checkpoint(cid)
