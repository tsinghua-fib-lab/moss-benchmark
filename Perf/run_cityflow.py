import json
import os
import time
from tempfile import NamedTemporaryFile

import cityflow
from common import parse_args
from tqdm import tqdm


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print('File already exists!')
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    cfg = {
        "interval": 1.0,
        "seed": 0,
        "dir": os.path.abspath("."),
        "roadnetFile": f"/data/{args.data}/cityflow/roadnet.json",
        "flowFile": f"/data/{args.data}/cityflow/flow.json",
        "rlTrafficLight": True,
        "saveReplay": False,
        "roadnetLogFile": "",
        "replayLogFile": "",
        "laneChange": False
    }
    with open(f"./data/{args.data}/cityflow/roadnet.json", 'r') as f:
        inter_list = json.load(f)['intersections']
    inters_dict = {}
    for inter in inter_list:
        inters_dict[inter['id']] = len(inter['trafficLight']['lightphases'])

    with NamedTemporaryFile('w', dir='.') as f:
        json.dump(cfg, f, indent=4)
        f.flush()
        eng = cityflow.Engine(config_file=f.name, thread_num=args.threads)

    log = []
    with tqdm(range(args.steps), ncols=100) as bar:
        for _ in bar:
            t = time.time()
            for inter_id in inters_dict:
                if inters_dict[inter_id] > 0:
                    eng.set_tl_phase(inter_id, ((int(eng.get_current_time()) // 30) % inters_dict[inter_id]))
            cnt = eng.get_vehicle_count()
            bar.set_description(f'veh: {cnt}')
            eng.next_step()
            log.append(time.time() - t)
    json.dump(log, open(args.output, 'w'))


if __name__ == '__main__':
    main()
