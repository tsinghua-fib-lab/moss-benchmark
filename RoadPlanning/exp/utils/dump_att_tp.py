import argparse
import pickle

from moss import Engine, TlPolicy, Verbosity


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", default="")
    parser.add_argument("--person_path", default="")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--tp_output_path", default="")
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
eng.set_tl_duration_batch(range(eng.junction_count), 30)  # type:ignore
eng.set_tl_policy_batch(
    range(eng.junction_count), TlPolicy.FIXED_TIME  # type:ignore
)
eng.next_step(n=3600 * 6)
att = eng.get_departed_person_average_traveling_time()
tp = eng.get_finished_person_count()
pickle.dump(att, open(args.output_path, "wb"))
pickle.dump(tp, open(args.tp_output_path, "wb"))
