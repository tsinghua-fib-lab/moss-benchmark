import argparse
import json
import os
import random
import time
from glob import glob

import numpy as np
from engine import get_engine
from moss import Engine
from moss.map import LaneType
from tqdm.auto import tqdm


class Env:
    def __init__(self, eng: Engine, data):
        self.eng = eng
        ps = json.load(open(f'{data}/road_pairs.json'))
        ps = sum(ps, [])
        self.rs = np.array(ps[0::3]).reshape(-1, 2)
        self.ls = np.array(ps[1::3]).reshape(-1, 2)
        self.ns = np.array(ps[2::3]).reshape(-1, 2)

    def get_vehicle_counts(self):
        cnt = self.eng.get_road_vehicle_counts()
        return cnt[self.rs]

    def set_state(self, states):
        """
        - `0` - disable
        - `1` - First Road
        - `2` - Second Road
        """
        assert len(states) == len(self.rs)
        for s, r, l in zip(states, self.rs, self.ls):
            if s == 0:
                self.eng.set_road_lane_plan(r[0], 0)
                self.eng.set_road_lane_plan(r[1], 0)
                self.eng.set_lane_restriction(l[0], True)
                self.eng.set_lane_restriction(l[1], True)
            elif s == 1:
                self.eng.set_road_lane_plan(r[0], 1)
                self.eng.set_road_lane_plan(r[1], 0)
                self.eng.set_lane_restriction(l[0], False)
                self.eng.set_lane_restriction(l[1], True)
            elif s == 2:
                self.eng.set_road_lane_plan(r[0], 0)
                self.eng.set_road_lane_plan(r[1], 1)
                self.eng.set_lane_restriction(l[0], True)
                self.eng.set_lane_restriction(l[1], False)
            else:
                assert False


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
        if glob(f'log/{args.algo}/{args.exp}/*/info.log'):
            return
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    eng = get_engine(
        map_file=f'{args.data}/map.bin',
        agent_file=f'{args.data}/agents.bin',
        start_step=args.start,
    )
    env = Env(eng, args.data)
    if args.algo == 'none':
        env.set_state([0]*len(env.rs))
    else:
        env.set_state([1]*len(env.rs))
    # M = eng.get_map()
    # reward = 0
    # road_lanes = [sorted(l.index for l in M.roads[r].lanes if l.type == LaneType.DRIVING) for r in env.rs.reshape(-1)]
    t = time.time()
    for _ in tqdm(range(args.steps//args.interval), ncols=90):
        if args.algo == 'none':
            pass
        elif args.algo == 'random':
            env.set_state([random.randint(0, 1) for _ in env.rs])
        elif args.algo == 'rule':
            cnt = env.get_vehicle_counts()
            n1 = env.ns.copy()
            n1[:, 1] -= 1
            n2 = env.ns.copy()
            n2[:, 0] -= 1
            c1 = np.max(cnt/n1, 1)
            c2 = np.max(cnt/n2, 1)
            env.set_state((c1 > c2)+1)
        else:
            raise NotImplementedError
        eng.next_step(args.interval)
    #     cnt = eng.get_lane_waiting_vehicle_counts()
    #     cnt = np.minimum(200, cnt)/200*5
    #     reward += np.mean([-np.mean(cnt[i]) for i in road_lanes])
    # print(f'{args.data}\t{args.algo}\tATT: {eng.get_departed_vehicle_average_traveling_time():.3f}\tTP: {eng.get_finished_vehicle_count()} Reward:{reward:.3f}')
    with open(f'{path}/info.log', 'a') as f:
        f.write(f"{eng.get_departed_vehicle_average_traveling_time():.3f} {eng.get_finished_vehicle_count()} {time.time()-t:.3f}\n")


main()
