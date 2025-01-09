import argparse
import os
import time

from engine import get_engine
from moss import TlPolicy
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument("--data", type=str, default="./data/us_newyork")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--device", type=int, default=0)

    args = parser.parse_args()

    if args.exp is None:
        path = time.strftime("log/ft/%Y%m%d-%H%M%S")
    else:
        path = time.strftime(f"log/ft/{args.exp}/%Y%m%d-%H%M%S")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    eng = get_engine(
        map_file=f"{args.data}/map.bin",
        person_file=f"{args.data}/agents.bin",
        start_step=args.start,
        device=args.device,
    )
    t = time.time()
    eng.set_tl_policy_batch([i for i in range(eng.junction_count)], TlPolicy.FIXED_TIME)
    total_step = args.steps
    interval = args.interval
    _to_run_step = [interval for _ in range(total_step // interval)]
    if total_step % interval > 0:
        _to_run_step.append(total_step % interval)
    for step in tqdm(_to_run_step):
        eng.next_step(step)
    with open(f"{path}/info.log", "a") as f:
        f.write(
            f"{eng.get_departed_person_average_traveling_time():.3f} {eng.get_finished_person_count()} {time.time()-t:.3f}\n"
        )


main()
