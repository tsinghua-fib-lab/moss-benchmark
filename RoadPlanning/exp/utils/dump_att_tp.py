import argparse
import pickle

from moss import Engine, LaneChange, TlPolicy, Verbosity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", default="")
    parser.add_argument("--agent_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--tp_output_path", default="")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--start_step", help="start step of the simulation", type=int, default=6*3600)
    return parser.parse_args()


args = get_args()
eng = Engine(
    map_file=args.map_path,
    agent_file=args.agent_path,
    start_step=args.start_step,
    lane_change=LaneChange.MOBIL,
    lane_veh_add_buffer_size=1500,
    lane_veh_remove_buffer_size=1500,
    verbose_level=Verbosity.INIT_ONLY,
    device=args.device_id,
)
eng.set_tl_duration_batch(range(eng.junction_count), 30)  # type:ignore
eng.set_tl_policy_batch(
    range(eng.junction_count), TlPolicy.FIXED_TIME  # type:ignore
)
eng.next_step(n=3600 * 6)
att = eng.get_departed_vehicle_average_traveling_time()
tp = eng.get_finished_vehicle_count()
pickle.dump(att, open(args.output_path, "wb"))
pickle.dump(tp, open(args.tp_output_path, "wb"))
