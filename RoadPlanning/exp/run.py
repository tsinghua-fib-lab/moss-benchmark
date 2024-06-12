import argparse
import asyncio
import logging
import os
import pickle
import random
import signal
import subprocess
import time

import numpy as np
import pandas as pd
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.general import GeneralBO
from mosstool.type import Map
from mosstool.util.format_converter import pb2dict
from utils.const import *
from utils.utils import (build_new_map, fetch_trip_route, get_geojson_data,
                         get_home_and_work_persons, show_best_result)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--city", default="")
    parser.add_argument("--opt_way_path", default="")
    parser.add_argument("--map_path", default="")
    parser.add_argument("--trip_path", default="")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--algo", choices=['none', 'random', 'rule', 'bo'])
    return parser.parse_args()


args = get_args()
path = time.strftime(f'log/{args.algo}/{args.exp}/%Y%m%d-%H%M%S')
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


async def main():
    if not os.path.exists("./LOOP_DATA/"):
        os.makedirs("./LOOP_DATA/", exist_ok=True)
    random.seed(0)
    logging.info(f"Processing road planning exp {args.algo.upper()} {args.city}!")
    opt_way_ids = pickle.load(open(args.opt_way_path, "rb"))
    with open(args.map_path, "rb") as f:
        ORIG_MAP = Map()
        ORIG_MAP.ParseFromString(f.read())
    orig_map_dict = pb2dict(ORIG_MAP)
    params = [
        {"name": "x" + str(i), "type": "int", "lb": 0, "ub": 1}
        for i in range(NUM_OF_BUILD_WAYS)
    ]
    space = DesignSpace().parse(params)
    general_bo = GeneralBO(space, num_obj=1, num_constr=1, model_config=BO_CONF)
    (orig_topo_dict, way_id2junc_id, _, _) = get_geojson_data(
        roadnet_path=f"./data/roadnet_{args.city}.geojson", proj_str=ORIG_MAP.header.projection
    )

    def obj(param: pd.DataFrame, att_ave) -> np.ndarray:
        o = np.array([att_ave for _ in range(1)]).reshape((1, 1))
        total_num = 0.0
        for ii in param.values[0]:
            total_num += ii
        c_ratio = (total_num / BUILD_WAYS_UPPER_BOUND) - 1
        c = np.array([np.exp(c_ratio) - 1]).reshape((1, 1))
        return np.hstack([o, c])

    ALL_ATTS = []
    ALL_PAIRS_VALUES = []
    ALL_TPS = []
    for _ in range(args.epochs):
        start_time = time.time()
        if args.algo == "bo":
            rec_x = general_bo.suggest()
            rec_x_values = rec_x.values[0]
        elif args.algo == "none":
            rec_x_values = [0 for _ in range(NUM_OF_BUILD_WAYS)]
        elif args.algo == "random":
            sum_of_res = 10 * NUM_OF_BUILD_WAYS
            rec_x_values = [0 for _ in range(NUM_OF_BUILD_WAYS)]
            while sum_of_res > BUILD_WAYS_UPPER_BOUND:
                rec_x_values = random.choices(
                    [
                        0,
                        1,
                    ],
                    k=NUM_OF_BUILD_WAYS,
                )
                sum_of_res = np.sum(rec_x_values)
        elif args.algo == "rule":
            rec_x_values = [
                0 for _ in range(NUM_OF_BUILD_WAYS - BUILD_WAYS_UPPER_BOUND)
            ] + [1 for _ in range(BUILD_WAYS_UPPER_BOUND)]
        else:
            raise NotImplementedError
        (new_m_dict, new_map_pb) = build_new_map(
            orig_topo_dict,
            ORIG_MAP,
            opt_way_ids,
            rec_x_values,  # type:ignore
            way_id2junc_id,
            args.city,
        )
        loop_map_name = f"moss.{args.city}_map"
        loop_map_file = f"./LOOP_DATA/{loop_map_name}.pb"
        loop_agent_file_to_work = f"./LOOP_DATA/to_work_{args.city}_trip.pb"
        loop_agent_file_to_home = f"./LOOP_DATA/to_home_{args.city}_trip.pb"
        loop_att_path = f"./LOOP_DATA/{args.city}_att.pkl"
        loop_tp_path = f"./LOOP_DATA/{args.city}_tp.pkl"
        with open(loop_map_file, "wb") as f:
            f.write(new_map_pb.SerializeToString())
        route_command = (
            f"./exp/utils/routing -map {loop_map_name} -cache ./LOOP_DATA -listen {HOST}"
        )
        cmd = route_command.split(" ")
        process = subprocess.Popen(args=cmd,)
        orig_person_path = args.trip_path
        (to_work_persons, to_home_persons, max_num) = get_home_and_work_persons(
            orig_person_path, orig_map_dict, new_m_dict
        )
        logging.info(
            f"to_work_persons {len(to_work_persons)}, to_home_persons {len(to_home_persons)}"
        )
        time.sleep(5)
        await fetch_trip_route(
            persons=to_home_persons,
            m_dict=new_m_dict,
            listen=HOST,
            output_path_home=loop_agent_file_to_home,
            output_path_work=loop_agent_file_to_work,
            max_num=max_num,
            other_persons=to_work_persons,
            transfer_to_lane_pos=False,
        )
        time.sleep(0.1)
        process.send_signal(sig=signal.SIGTERM)
        process.wait()
        # morning peak 6-12
        cmd = f"python3 exp/utils/dump_att_tp.py --map_path {loop_map_file} --agent_path {loop_agent_file_to_work} --tp_output_path {loop_tp_path} --output_path {loop_att_path} --device_id {args.device_id}".split(
            " "
        )
        subprocess.run(cmd, check=True)
        att_0 = pickle.load(open(loop_att_path, "rb"))
        tp_0 = pickle.load(open(loop_tp_path, "rb"))
        # evening peak 17-23
        cmd = f"python3 exp/utils/dump_att_tp.py --map_path {loop_map_file} --start_step {17*3600} --agent_path {loop_agent_file_to_home} --tp_output_path {loop_tp_path} --output_path {loop_att_path} --device_id {args.device_id}".split(
            " "
        )
        subprocess.run(cmd, check=True)
        att_1 = pickle.load(open(loop_att_path, "rb"))
        tp_1 = pickle.load(open(loop_tp_path, "rb"))
        tp_ave = np.mean([tp_0, tp_1])
        att_ave = np.mean([att_0, att_1])
        ALL_TPS.append(tp_ave)
        ALL_ATTS.append(att_ave)
        ALL_PAIRS_VALUES.append([v for v in rec_x_values])  # type:ignore
        if args.algo == "bo":
            general_bo.observe(rec_x, obj(rec_x, att_ave))  # type:ignore
        end_time = time.time()
        logging.info(f"iter {len(ALL_ATTS)-1}:{(end_time - start_time)}")
        with open(f'{path}/info.log', 'a') as f:
            f.write(f"{att_ave:.3f} {tp_ave:.3f} {end_time - start_time:.3f}\n")

    best_att, best_tp = show_best_result(
        atts=ALL_ATTS, tps=ALL_TPS, algo=args.algo
    )
    logging.info(f"Best res: ATT {best_att} TP: {best_tp}")


if __name__ == "__main__":
    asyncio.run(main())
