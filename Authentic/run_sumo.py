import json
import os
import time
from tempfile import NamedTemporaryFile

import traci
from common import parse_args
from tqdm import tqdm

os.environ['SUMO_HOME'] = '/usr/share/sumo'


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print('File already exists!')
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    try:
        with NamedTemporaryFile('w', dir='.') as f:
            f.write(f'''<configuration>
<input>
<net-file value="./data/{args.data}/sumo/roadnet.net.xml"/>
<route-files value="./data/{args.data}/sumo/flow.rou.xml"/>
</input>
<time>
<begin value="0"/>
<end value="{args.steps}"/>
</time>
</configuration>''')
            f.flush()
            traci.start(f'/usr/bin/sumo -c {f.name}'.split())
        ts_id_list = traci.trafficlight.getIDList()

        log = []
        with tqdm(range(args.steps), ncols=100) as bar:
            for _ in bar:
                t = time.time()
                for ts_id in ts_id_list:
                    if int(traci.trafficlight.getNextSwitch(ts_id)) >= 1:
                        traci.trafficlight.setPhase(ts_id, traci.trafficlight.getPhase(ts_id))
                cnt = traci.simulation.getMinExpectedNumber()
                bar.set_description(f'veh: {cnt}')
                traci.simulationStep()
                log.append(time.time() - t)
    finally:
        traci.close()
    json.dump(log, open(args.output, 'w'))


if __name__ == '__main__':
    main()
