import argparse
import os
import time

from engine import get_engine
from moss import TlPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument('--data', type=str, default='./data/us_newyork')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)

    args = parser.parse_args()

    if args.exp is None:
        path = time.strftime('log/ft/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/ft/{args.exp}/%Y%m%d-%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    eng = get_engine(
        map_file=f'{args.data}/map.bin',
        agent_file=f'{args.data}/agents.bin',
        start_step=args.start,
    )
    t = time.time()
    eng.set_tl_policy_batch(range(eng.junction_count), TlPolicy.FIXED_TIME)
    eng.next_step(args.steps)
    with open(f'{path}/info.log', 'a') as f:
        f.write(f"{eng.get_departed_vehicle_average_traveling_time():.3f} {eng.get_finished_vehicle_count()} {time.time()-t:.3f}\n")


main()
