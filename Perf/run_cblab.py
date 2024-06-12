import json
import os
import time
from tempfile import NamedTemporaryFile

import cbengine
from common import parse_args
from tqdm import tqdm


class Wrapper(object):
    '''
    NOTE: this class is copied from the original code of CBLab
    '''

    def __init__(self, roadnet_file):
        # refer to cbengine/env/CBEngine/envs/CBEngine.py
        # here agent is those intersections with signals
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.lane_vehicle_state = {}
        self.log_enable = 1
        self.warning_enable = 1
        self.ui_enable = 1
        self.info_enable = 1
        with open(roadnet_file, 'r') as f:
            lines = f.readlines()
            cnt = 0
            pre_road = 0
            is_obverse = 0
            for line in lines:
                line = line.rstrip('\n').split(' ')
                if ('' in line):
                    line.remove('')
                if (len(line) == 1):  # the notation line
                    if (cnt == 0):
                        self.agent_num = int(line[0])  # start the intersection segment
                        cnt += 1
                    elif (cnt == 1):
                        self.road_num = int(line[0]) * 2  # start the road segment
                        cnt += 1
                    elif (cnt == 2):
                        self.signal_num = int(line[0])  # start the signal segment
                        cnt += 1
                else:
                    if (cnt == 1):  # in the intersection segment
                        self.intersections[int(line[2])] = {
                            'latitude': float(line[0]),
                            'longitude': float(line[1]),
                            'have_signal': int(line[3]),
                            'end_roads': [],
                            'start_roads': []
                        }
                    elif (cnt == 2):  # in the road segment
                        if (len(line) != 8):
                            road_id = pre_road[is_obverse]
                            self.roads[road_id]['lanes'] = {}
                            for i in range(self.roads[road_id]['num_lanes']):
                                self.roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                                self.lane_vehicle_state[road_id * 100 + i] = set()
                            is_obverse ^= 1
                        else:
                            self.roads[int(line[-2])] = {
                                'start_inter': int(line[0]),
                                'end_inter': int(line[1]),
                                'length': float(line[2]),
                                'speed_limit': float(line[3]),
                                'num_lanes': int(line[4]),
                                'inverse_road': int(line[-1])
                            }
                            self.roads[int(line[-1])] = {
                                'start_inter': int(line[1]),
                                'end_inter': int(line[0]),
                                'length': float(line[2]),
                                'speed_limit': float(line[3]),
                                'num_lanes': int(line[5]),
                                'inverse_road': int(line[-2])
                            }
                            self.intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                            self.intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                            self.intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                            self.intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                            pre_road = (int(line[-2]), int(line[-1]))
                    else:
                        # 4 out-roads
                        signal_road_order = list(map(int, line[1:]))
                        now_agent = int(line[0])
                        in_roads = []
                        for road in signal_road_order:
                            if (road != -1):
                                in_roads.append(self.roads[road]['inverse_road'])
                            else:
                                in_roads.append(-1)
                        in_roads += signal_road_order
                        self.agents[now_agent] = in_roads

                        # 4 in-roads
                        # self.agents[int(line[0])] = self.intersections[int(line[0])]['end_roads']
                        # 4 in-roads plus 4 out-roads
                        # self.agents[int(line[0])] += self.intersections[int(line[0])]['start_roads']
        for agent, agent_roads in self.agents.items():
            self.intersections[agent]['lanes'] = []
            for road in agent_roads:
                # here we treat road -1 have 3 lanes
                if (road == -1):
                    for i in range(3):
                        self.intersections[agent]['lanes'].append(-1)
                else:
                    for lane in self.roads[road]['lanes'].keys():
                        self.intersections[agent]['lanes'].append(lane)


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.overwrite:
        print('File already exists!')
        return
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with NamedTemporaryFile('w', dir='.') as f:
        f.write(f'''start_time_epoch = 0
max_time_epoch = {args.steps}
road_file_addr : ./data/{args.data}/cblab/roadnet.txt
vehicle_file_addr : ./data/{args.data}/cblab/flow.txt
report_log_mode : nolog
report_log_addr : ./
report_log_rate = 1
warning_stop_time_log = 10000''')
        f.flush()
        eng = cbengine.Engine(f.name, 20)

    wp = Wrapper(f'./data/{args.data}/cblab/roadnet.txt')

    log = []
    with tqdm(range(args.steps), ncols=100) as bar:
        for _ in bar:
            t = time.time()
            for intersection in wp.intersections:
                eng.set_ttl_phase(intersection, ((int(eng.get_current_time()) // 30) % 4) + 1)
            cnt = eng.get_vehicle_count()
            bar.set_description(f'veh: {cnt}')
            eng.next_step()
            log.append(time.time() - t)
    json.dump(log, open(args.output, 'w'))


if __name__ == '__main__':
    main()
