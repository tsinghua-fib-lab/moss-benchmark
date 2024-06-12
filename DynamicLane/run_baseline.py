import argparse
import os
import random
import time

import numpy as np
from engine import get_engine
from moss.map import LaneType
from tqdm.auto import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument('--data', type=str, default='data/us_newyork')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=7200)
    parser.add_argument('--interval', type=int, default=180)
    parser.add_argument('--algo', choices=['none', 'random', 'rule'], required=True)
    parser.add_argument('--seed', type=int, default=43)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.exp is None:
        path = time.strftime(f'log/{args.algo}/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/{args.algo}/{args.exp}/%Y%m%d-%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    eng = get_engine(
        map_file=f'{args.data}/map.bin',
        agent_file=f'{args.data}/agents.bin',
        start_step=args.start,
    )
    M = eng.get_map()
    r_ids = [r.index for r in M.roads if len(r.pb.next_road_lane_plans) > 1]
    if args.algo == 'rule':
        nrl = [
            [
                [list(range(M.lane_map[l.lane_id_a].index, M.lane_map[l.lane_id_b].index+1)) for l in p.next_road_lanes]
                for p in M.roads[r].pb.next_road_lane_plans
            ]
            for r in r_ids
        ]
    r_plan_ids = [0]*len(r_ids)
    reward = 0
    road_lanes = [sorted(l.index for l in M.roads[r].lanes if l.type == LaneType.DRIVING) for r in r_ids]
    t = time.time()
    for _ in tqdm(range(args.steps//args.interval), ncols=90):
        if args.algo == 'none':
            pass
        elif args.algo == 'random':
            r_plan_ids = [random.randint(0, 1) for _ in r_plan_ids]
        elif args.algo == 'rule':
            cnt = eng.get_lane_waiting_at_end_vehicle_counts()
            new_plan = []
            for nr, i in zip(nrl, r_plan_ids):
                c = [cnt[x].sum() for x in nr[i]]
                ps = []
                for i, p in enumerate(nr):
                    p = np.array([x/len(y) for x, y in zip(c, p)])
                    p = max(p)
                    ps.append(p)
                new_plan.append(np.argmin(ps))
            r_plan_ids[:] = new_plan
            # print(''.join(map(str, r_plan_ids)))
        else:
            raise NotImplementedError
        eng.set_road_lane_plan_batch(r_ids, r_plan_ids)
        eng.next_step(args.interval)
        cnt = eng.get_lane_waiting_vehicle_counts()
        cnt = np.minimum(200, cnt)/200*5
        reward += np.mean([-np.mean(cnt[i]) for i in road_lanes])
    print(f'{args.algo}\tATT: {eng.get_departed_vehicle_average_traveling_time():.3f}\tTP: {eng.get_finished_vehicle_count()} Reward:{reward:.3f}')
    with open(f'{path}/info.log', 'a') as f:
        f.write(f"{eng.get_departed_vehicle_average_traveling_time():.3f} {eng.get_finished_vehicle_count()} {time.time()-t:.3f}\n")


main()
