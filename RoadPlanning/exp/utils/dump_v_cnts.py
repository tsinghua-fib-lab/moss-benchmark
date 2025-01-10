import argparse
import pickle

import numpy as np
from moss import Engine, TlPolicy, Verbosity
from moss_engine import MossApiEngine


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
moss_eng = Engine(
    name=f"RoadPlanning",
    map_file=args.map_path,
    person_file=args.person_path,
    start_step=args.start_step,
    verbose_level=Verbosity.INIT_ONLY,
    device=args.device_id,
)
eng = MossApiEngine(moss_eng)
moss_eng.set_tl_duration_batch([i for i in range(eng.junction_count)], 30)
moss_eng.set_tl_policy_batch(
    [i for i in range(eng.junction_count)], TlPolicy.FIXED_TIME
)
all_v_cnts: list[np.ndarray] = []
for _ in range(int((3600 * 6 + 1 - 1) / 300)):
    eng.next_step(n=300)
    all_v_cnts.append(eng.get_road_vehicle_counts())
all_v_cnts_array = np.array(all_v_cnts)
ave_v_cnts = np.mean(np.abs(all_v_cnts_array), axis=0)
pickle.dump(ave_v_cnts, open(args.output_path, "wb"))
