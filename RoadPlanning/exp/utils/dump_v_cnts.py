import argparse
import pickle

import numpy as np
from moss import Engine, TlPolicy, Verbosity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", default="")
    parser.add_argument("--person_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--start_step", help="start step of the simulation", type=int, default=6 * 3600
    )
    return parser.parse_args()


args = get_args()
eng = Engine(
    name=f"RoadPlanning",
    map_file=args.map_path,
    person_file=args.person_path,
    start_step=args.start_step,
    verbose_level=Verbosity.INIT_ONLY,
    device=args.device_id,
)
all_road_ids = [i + 2_0000_0000 for i in range(eng.road_count)]
eng.set_tl_duration_batch([i for i in range(eng.junction_count)], 30)
eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.FIXED_TIME)
all_v_cnts: list[list[int]] = []
for _ in range(int((3600 * 6 + 1 - 1) / 300)):
    eng.next_step(n=300)
    _road_id2cnt: dict[int, int] = eng.get_road_vehicle_counts()
    all_v_cnts.append([_road_id2cnt.get(road_id,0) for road_id in all_road_ids])
all_v_cnts_array = np.array(all_v_cnts)
ave_v_cnts = np.mean(np.abs(all_v_cnts_array), axis=0)
pickle.dump(ave_v_cnts, open(args.output_path, "wb"))
