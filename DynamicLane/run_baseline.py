import argparse
import os
import random
import time

import numpy as np
from engine import get_engine
from mosstool.type import Lane, LaneType, Map
from tqdm.auto import tqdm

ROAD_ID_START = 2_0000_0000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument("--data", type=str, default="data/us_newyork")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--steps", type=int, default=7200)
    parser.add_argument("--interval", type=int, default=180)
    parser.add_argument("--algo", choices=["none", "random", "rule"], required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.exp is None:
        path = time.strftime(f"log/{args.algo}/%Y%m%d-%H%M%S")
    else:
        path = time.strftime(f"log/{args.algo}/{args.exp}/%Y%m%d-%H%M%S")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    eng = get_engine(
        map_file=f"{args.data}/map.bin",
        person_file=f"{args.data}/agents.bin",
        start_step=args.start,
    )
    M: Map = eng.get_map(dict_return=False)  # type:ignore
    map_lanes_dict: dict[int, Lane] = {i.id: i for i in M.lanes}
    all_lane_ids: list[int] = [i for i in range(eng.lane_count)]
    all_road_ids: list[int] = [i + ROAD_ID_START for i in range(eng.road_count)]
    lane_map = {lid: idx for idx, lid in enumerate(all_lane_ids)}
    road_map = {rid: idx for idx, rid in enumerate(all_road_ids)}
    r_ids = [road_map[r.id] for r in M.roads if len(r.next_road_lane_plans) > 1]
    if args.algo == "rule":
        nrl = [
            [
                [
                    list(range(lane_map[l.lane_id_a], lane_map[l.lane_id_b] + 1))
                    for l in p.next_road_lanes
                ]
                for p in M.roads[r].next_road_lane_plans
            ]
            for r in r_ids
        ]
    else:
        nrl = []
    r_plan_ids = [0] * len(r_ids)
    reward = 0
    road_lanes = [
        sorted(
            lane_map[lid]
            for lid in M.roads[r].lane_ids
            if map_lanes_dict[lid].type == LaneType.LANE_TYPE_DRIVING
        )
        for r in r_ids
    ]
    t = time.time()
    for _ in tqdm(range(args.steps // args.interval), ncols=90):
        if args.algo == "none":
            pass
        elif args.algo == "random":
            r_plan_ids = [random.randint(0, 1) for _ in r_plan_ids]
        elif args.algo == "rule":
            cnt_dict = eng.get_lane_waiting_at_end_vehicle_counts()
            cnt = np.array([cnt_dict[lid] for lid in all_lane_ids])
            new_plan = []
            for nr, i in zip(nrl, r_plan_ids):
                c = [cnt[x].sum() for x in nr[i]]
                ps = []
                for i, p in enumerate(nr):
                    p = np.array([x / len(y) for x, y in zip(c, p)])
                    p = max(p)
                    ps.append(p)
                new_plan.append(np.argmin(ps))
            r_plan_ids[:] = new_plan
            # print(''.join(map(str, r_plan_ids)))
        else:
            raise NotImplementedError
        eng.set_road_lane_plan_batch(r_ids, r_plan_ids)
        eng.next_step(args.interval)
        cnt_dict = eng.get_lane_waiting_vehicle_counts()
        cnt = np.array([cnt_dict[lid] for lid in all_lane_ids])
        cnt = np.minimum(200, cnt) / 200 * 5
        reward += np.mean([-np.mean(cnt[i]) for i in road_lanes])
    print(
        f"{args.algo}\tATT: {eng.get_departed_person_average_traveling_time():.3f}\tTP: {eng.get_finished_person_count()} Reward:{reward:.3f}"
    )
    with open(f"{path}/info.log", "a") as f:
        f.write(
            f"{eng.get_departed_person_average_traveling_time():.3f} {eng.get_finished_person_count()} {time.time()-t:.3f}\n"
        )


main()
