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
moss_eng = Engine(
    name=f"RoadPlanning",
    map_file=args.map_path,
    person_file=args.person_path,
    start_step=args.start_step,
    verbose_level=Verbosity.INIT_ONLY,
    device=args.device_id,
)
moss_eng.set_tl_duration_batch([i for i in range(moss_eng.junction_count)], 30)
moss_eng.set_tl_policy_batch(
    [i for i in range(moss_eng.junction_count)], TlPolicy.FIXED_TIME
)
moss_eng.next_step(n=3600 * 6)
att = moss_eng.get_departed_person_average_traveling_time()
tp = moss_eng.get_finished_person_count()
pickle.dump(att, open(args.output_path, "wb"))
pickle.dump(tp, open(args.tp_output_path, "wb"))
