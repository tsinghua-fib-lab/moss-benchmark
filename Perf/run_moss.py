import json
import os
import time

from common import parse_args
from moss import Engine
from tqdm import tqdm


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print('File already exists!')
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    t = time.time()
    eng = Engine(
        map_file=f'data/{args.data}/moss/map.bin',
        agent_file=f'data/{args.data}/moss/agents.bin',
        disable_aoi_out_control=True
    )
    log = []
    with tqdm(range(args.steps)) as bar:
        for _ in bar:
            t = time.time()
            bar.set_description(f'veh: {eng.get_running_vehicle_count()}')
            eng.next_step()
            log.append(time.time() - t)
    json.dump(log, open(args.output, 'w'))


if __name__ == '__main__':
    main()
