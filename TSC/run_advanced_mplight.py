import argparse
import json
import os
import random
import sys
import time
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from engine import get_engine
from moss.map import LaneTurn, LaneType, LightState
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out


class Env:
    def __init__(self, data_path, start_step, step_size, step_count, log_dir, reward, alpha=0):
        self.log_dir = log_dir
        self.eng = get_engine(
            map_file=f'{data_path}/map.bin',
            agent_file=f'{data_path}/agents.bin',
            start_step=start_step,
        )
        self.alpha = alpha
        M = self.eng.get_map()

        def lanes_collect(js):
            in_lanes_list = []
            out_lanes_list = []
            phase_lanes_list = []
            phase_lanes_A_list, phase_lanes_B_list = [], []
            phase_label_list = []

            for jid in js:
                junction = M.junction_map[jid]
                phases_lane = []
                phases_lanes_A, phases_lanes_B = [], []
                in_lane, out_lane = [], []
                in_lane_A, in_lane_B, out_lane_A, out_lane_B = [], [], [], []
                labels = []
                if junction.tl:
                    tl = junction.tl
                    for phase in tl.phases:
                        lanes = [i for i, j in zip(junction.lanes, phase.states) if j == LightState.GREEN and i.type == LaneType.DRIVING and i.turn != LaneTurn.RIGHT and i.turn != LaneTurn.AROUND]
                        in_lanes = [m.predecessors[0].id for m in lanes]
                        out_lanes = [m.successors[0].id for m in lanes]
                        phases_lane.append([list(set(in_lanes)), list(set(out_lanes))])
                        in_lane += in_lanes
                        out_lane += out_lanes
                        labels.append([
                            any(i.turn == LaneTurn.STRAIGHT for i in lanes),
                            any(i.turn == LaneTurn.LEFT for i in lanes)
                        ])
                        # 对lanes根据具体的travel movement进行分类
                        in_angles = [lane.geom.angle_in for lane in lanes]
                        in_angles = np.array(in_angles)-min(in_angles)
                        lanes_tmB = (in_angles >= np.pi/2) & (in_angles <= 3*np.pi/2)
                        lanes_tmA = [not i for i in lanes_tmB]
                        lanes_tmA, lanes_tmB = np.array(lanes)[lanes_tmA], np.array(lanes)[lanes_tmB]
                        in_lane_A = [m.predecessors[0].id for m in lanes_tmA]
                        in_lane_B = [m.predecessors[0].id for m in lanes_tmB]
                        out_lane_A = [m.successors[0].id for m in lanes_tmA]
                        out_lane_B = [m.successors[0].id for m in lanes_tmB]
                        phases_lanes_A.append([list(set(in_lane_A)), list(set(out_lane_A))])
                        phases_lanes_B.append([list(set(in_lane_B)), list(set(out_lane_B))])
                in_lanes_list.append(list(set(in_lane)))
                out_lanes_list.append(list(set(out_lane)))
                phase_lanes_list.append(phases_lane)
                phase_lanes_A_list.append(phases_lanes_A)
                phase_lanes_B_list.append(phases_lanes_B)
                phase_label_list.append(labels)
            return in_lanes_list, out_lanes_list, phase_lanes_list, phase_label_list, phase_lanes_A_list, phase_lanes_B_list

        self.jids = [i for i, j in enumerate(self.eng.get_junction_phase_counts()) if j > 1]
        js = [M.junctions[i].id for i in self.jids]
        self.in_lanes, self.out_lanes, self.jpl, self.jpl_label, self.jpl_A, self.jpl_B = lanes_collect(js)

        def in_lane_numpy(in_lanes):
            max_in_lane_num = max([len(i) for i in in_lanes])
            in_lanes_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            in_lanes_zero_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            for i, in_lane in enumerate(in_lanes):
                in_lanes_array[i, :len(in_lane)] = in_lane
                in_lanes_zero_array[i, len(in_lane):] = 1
            return in_lanes_array, in_lanes_zero_array

        def phase_lane_numpy(phase_lanes_A, phase_lanes_B):
            max_inflow_lane_num_A = max_outflow_lane_num_A = max(max([max([len(j[1]) for j in i]) for i in phase_lanes_A]),  max([max([len(j[0]) for j in i]) for i in phase_lanes_A]))
            max_inflow_lane_num_B = max_outflow_lane_num_B = max(max([max([len(j[1]) for j in i]) for i in phase_lanes_B]),  max([max([len(j[0]) for j in i]) for i in phase_lanes_B]))
            phase_lanes_inflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_inflow_lane_num_A), dtype=int)
            phase_lanes_outflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_outflow_lane_num_A), dtype=int)
            zero_lanes_inflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_inflow_lane_num_A), dtype=int)
            zero_lanes_outflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_outflow_lane_num_A), dtype=int)
            non_zeros_inflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_inflow_lane_num_A), dtype=int)
            non_zeros_outflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_outflow_lane_num_A), dtype=int)
            missing_lanes_inflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_inflow_lane_num_A), dtype=int)
            missing_lanes_outflow_A = np.zeros((len(phase_lanes_A), self.max_action_size, max_outflow_lane_num_A), dtype=int)

            phase_lanes_inflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_inflow_lane_num_B), dtype=int)
            phase_lanes_outflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_outflow_lane_num_B), dtype=int)
            zero_lanes_inflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_inflow_lane_num_B), dtype=int)
            zero_lanes_outflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_outflow_lane_num_B), dtype=int)
            non_zeros_inflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_inflow_lane_num_B), dtype=int)
            non_zeros_outflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_outflow_lane_num_B), dtype=int)
            missing_lanes_inflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_inflow_lane_num_B), dtype=int)
            missing_lanes_outflow_B = np.zeros((len(phase_lanes_B), self.max_action_size, max_outflow_lane_num_B), dtype=int)

            missing_phase = np.zeros((len(phase_lanes_A), self.max_action_size), dtype=int)

            for i, phase_lane in enumerate(phase_lanes_A):
                for j, lanes in enumerate(phase_lane):
                    phase_lanes_inflow_A[i, j, :len(lanes[0])] = lanes[0]
                    phase_lanes_outflow_A[i, j, :len(lanes[1])] = lanes[1]
                    zero_lanes_inflow_A[i, j, len(lanes[0]):] = 1
                    zero_lanes_outflow_A[i, j, len(lanes[1]):] = 1
                    non_zeros_inflow_A[i, j, :len(lanes[0])] = 1
                    non_zeros_outflow_A[i, j, :len(lanes[1])] = 1
                if len(phase_lane) < self.max_action_size:
                    for j in range(len(phase_lane), self.max_action_size):
                        missing_lanes_inflow_A[i, j, :] = 1
                        missing_lanes_outflow_A[i, j, :] = 1
                        non_zeros_inflow_A[i, j, :] = 1
                        non_zeros_outflow_A[i, j, :] = 1
                        missing_phase[i, j] = 1

            for i, phase_lane in enumerate(phase_lanes_B):
                for j, lanes in enumerate(phase_lane):
                    phase_lanes_inflow_B[i, j, :len(lanes[0])] = lanes[0]
                    phase_lanes_outflow_B[i, j, :len(lanes[1])] = lanes[1]
                    zero_lanes_inflow_B[i, j, len(lanes[0]):] = 1
                    zero_lanes_outflow_B[i, j, len(lanes[1]):] = 1
                    non_zeros_inflow_B[i, j, :len(lanes[0])] = 1
                    non_zeros_outflow_B[i, j, :len(lanes[1])] = 1
                if len(phase_lane) < self.max_action_size:
                    for j in range(len(phase_lane), self.max_action_size):
                        missing_lanes_inflow_B[i, j, :] = 1
                        missing_lanes_outflow_B[i, j, :] = 1
                        non_zeros_inflow_B[i, j, :] = 1
                        non_zeros_outflow_B[i, j, :] = 1
            return phase_lanes_inflow_A, phase_lanes_outflow_A, zero_lanes_inflow_A, zero_lanes_outflow_A, non_zeros_inflow_A, non_zeros_outflow_A, missing_lanes_inflow_A, missing_lanes_outflow_A, phase_lanes_inflow_B, phase_lanes_outflow_B, zero_lanes_inflow_B, zero_lanes_outflow_B, non_zeros_inflow_B, non_zeros_outflow_B, missing_lanes_inflow_B, missing_lanes_outflow_B, missing_phase

        self.action_sizes = list(self.eng.get_junction_phase_counts())
        self.action_sizes = [self.action_sizes[i] for i in self.jids]
        self.max_action_size = max(self.action_sizes)
        self.available_actions = np.zeros((len(self.action_sizes), max(self.action_sizes)))
        for i, j in enumerate(self.action_sizes):
            self.available_actions[i, :j] = 1
        self.phase_lanes = self.jpl
        self.phase_lanes_A = self.jpl_A
        self.phase_lanes_B = self.jpl_B
        self.phase_lane_length = [[[M.lanes[lanes[0][0]].length, M.lanes[lanes[1][0]].length] for lanes in phase_lanes] for phase_lanes in self.phase_lanes]
        self.junction_type_list = [np.array([(m == 3) | (m == 6), (m == 4) | (m == 8)]).astype(int) for m in self.action_sizes]
        self.junction_scale_list = np.array([[len(i)/j] for i, j in zip(self.in_lanes, self.action_sizes)])

        self.phase_lanes_inflow_A, self.phase_lanes_outflow_A, self.zero_lanes_inflow_A, self.zero_lanes_outflow_A, self.non_zeros_inflow_A, self.non_zeros_outflow_A, self.missing_lanes_inflow_A, self.missing_lanes_outflow_A, self.phase_lanes_inflow_B, self.phase_lanes_outflow_B, self.zero_lanes_inflow_B, self.zero_lanes_outflow_B, self.non_zeros_inflow_B, self.non_zeros_outflow_B, self.missing_lanes_inflow_B, self.missing_lanes_outflow_B, self.missing_phase = phase_lane_numpy(
            self.phase_lanes_A, self.phase_lanes_B)
        self.in_lane_array, self.zero_in_lane_array = in_lane_numpy(self.in_lanes)

        self.junction_phase_sizes = np.zeros((len(self.jids), max(self.action_sizes) + 1))
        for i, phases in enumerate(self.phase_lanes):
            for j, phase in enumerate(phases):
                self.junction_phase_sizes[i, j] = len(phase[0])
            self.junction_phase_sizes[i, -1] = len(phases)

        self.junction_graph_edges, self.junction_graph_edge_feats = [], []
        self.junction_src_list = [[] for _ in range(len(self.jids))]
        for src_idx, out_lanes in tqdm(enumerate(self.out_lanes)):
            for out_lane in out_lanes:
                ids = [jid for (jid, in_lanes) in enumerate(self.in_lanes) if out_lane in in_lanes]
                if len(ids) != 0:
                    if [src_idx, ids[0]] not in self.junction_graph_edges:
                        self.junction_graph_edges.append([src_idx, ids[0]])
                        self.junction_graph_edge_feats.append([M.lanes[out_lane].length, 1])
                        self.junction_src_list[ids[0]].append(src_idx)
                    else:
                        idx = [i for i, value in enumerate(self.junction_graph_edges) if value == [src_idx, ids[0]]][0]
                        self.junction_graph_edge_feats[idx][1] += 1
        self.max_neighbor_num = max([len(i) for i in self.junction_src_list])
        print('Max Neighbor Number:', self.max_neighbor_num)
        self.connect_matrix = np.zeros((len(self.jids), len(self.jids)))
        for i, j in self.junction_graph_edges:
            self.connect_matrix[i, j] = 1

        # 标注邻居间的关系
        self.phase_relation = np.zeros((len(self.jids), self.max_neighbor_num, 2), dtype=int)  # one-hot type information
        self.neighbor_type = np.zeros((len(self.jids), self.max_neighbor_num), dtype=int)

        for dst_idx in range(len(self.jids)):
            src_idxs = self.junction_src_list[dst_idx]
            if len(src_idxs) == 0:
                continue
            dst_phase_lanes = self.phase_lanes[dst_idx]
            dst_in_lanes = self.in_lanes[dst_idx]
            for idx, src_idx in enumerate(src_idxs):
                src_phase_lanes = self.phase_lanes[src_idx]
                src_out_lanes = self.out_lanes[src_idx]
                for dst_phase_idx, dst_phase in enumerate(dst_phase_lanes):
                    if set(dst_phase[0]) & set(src_out_lanes) == set():
                        continue
                    if len(src_phase_lanes) > 2:
                        if set(src_phase_lanes[2][1]) & set(dst_in_lanes) != set() or set(src_phase_lanes[1][1]) & set(dst_in_lanes) != set():
                            self.phase_relation[dst_idx, idx, :] = [1, 0]
                        else:
                            self.phase_relation[dst_idx, idx, :] = [0, 1]
                    elif len(src_phase_lanes) == 2:
                        if set(src_phase_lanes[1][1]) & set(dst_in_lanes) != set():
                            self.phase_relation[dst_idx, idx, :] = [1, 0]
                        else:
                            self.phase_relation[dst_idx, idx, :] = [0, 1]
                    if dst_phase_idx in [0, 1]:
                        self.neighbor_type[dst_idx, idx] = 0
                    else:
                        self.neighbor_type[dst_idx, idx] = 1

        self.edge_distance = np.zeros((len(self.jids), self.max_neighbor_num, 1))
        self.edge_strength = np.zeros((len(self.jids), self.max_neighbor_num, 1))

        for i, idxs in enumerate(self.junction_src_list):
            if len(idxs) == 0:
                continue
            self.edge_distance[i, :len(idxs), :] = np.array([self.junction_graph_edge_feats[self.junction_graph_edges.index([j, i])][0] for j in idxs]).reshape(-1, 1)
            self.edge_strength[i, :len(idxs), :] = np.array([self.junction_graph_edge_feats[self.junction_graph_edges.index([j, i])][1] for j in idxs]).reshape(-1, 1)

        max_edge_distance = np.max(self.edge_distance)
        self.edge_distance = self.edge_distance/max_edge_distance

        self.edge_distance = np.concatenate([self.edge_distance, self.edge_strength], axis=2)

        print(f'Training on {len(self.jids)} junctions')
        self._cid = self.eng.make_checkpoint()
        # self.three_phases_idxs = [i for i, j in enumerate(self.action_sizes) if j == 3]
        # self.four_phases_idxs = [i for i, j in enumerate(self.action_sizes) if j == 4]
        self.step_size = step_size
        self.step_count = step_count
        self._step = 0
        self.reward = reward
        self.info = {
            'ATT': 1e999,
            'Throughput': 0,
            'reward': 0,
            'ATT_inside': 1e999,
            'ATT_finished': 1e999,
            'Throughput_inside': 0,
        }
        self.data_path = data_path
        self.one_hot_mapping_matrix = np.eye(self.max_action_size)

    def reset(self):
        self.eng.restore_checkpoint(self._cid)

    def observe(self):
        cnt = self.eng.get_lane_waiting_vehicle_counts()
        cnt_all = self.eng.get_lane_waiting_at_end_vehicle_counts(1e9, 150)
        cnt_running = cnt_all - cnt  # 在150米的道路阈值内在奔跑的车辆

        # Efficient pressure
        in_cnt_states_A, out_cnt_states_A = cnt[self.phase_lanes_inflow_A], cnt[self.phase_lanes_outflow_A]
        in_cnt_states_A[self.zero_lanes_inflow_A == 1] = 0
        out_cnt_states_A[self.zero_lanes_outflow_A == 1] = 0

        def div(a, b):
            mask = b == 0
            x = a/(b+1e-6)
            x[mask] = 0
            return x

        in_pressure_A = div(
            np.sum(in_cnt_states_A, axis=2),
            np.sum(self.non_zeros_inflow_A, axis=2)
        )
        out_pressure_A = div(
            np.sum(out_cnt_states_A, axis=2),
            np.sum(self.non_zeros_outflow_A, axis=2)
        )
        pressure_A = in_pressure_A - out_pressure_A
        pressure_A[self.missing_phase == 1] = -1e3

        in_cnt_states_B, out_cnt_states_B = cnt[self.phase_lanes_inflow_B], cnt[self.phase_lanes_outflow_B]
        in_cnt_states_B[self.zero_lanes_inflow_B == 1] = 0
        out_cnt_states_B[self.zero_lanes_outflow_B == 1] = 0

        in_pressure_B = div(
            np.sum(in_cnt_states_B, axis=2),
            np.sum(self.non_zeros_inflow_B, axis=2)
        )
        out_pressure_B = div(
            np.sum(out_cnt_states_B, axis=2),
            np.sum(self.non_zeros_outflow_B, axis=2)
        )
        pressure_B = in_pressure_B - out_pressure_B
        pressure_B[self.missing_phase == 1] = -1e3

        # Effective running vehicle number
        running_A = cnt_running[self.phase_lanes_inflow_A]
        running_A[self.zero_lanes_inflow_A == 1] = 0
        running_A = np.sum(running_A, axis=2)
        running_A[self.missing_phase == 1] = -1

        running_B = cnt_running[self.phase_lanes_inflow_B]
        running_B[self.zero_lanes_inflow_B == 1] = 0
        running_B = np.sum(running_B, axis=2)
        running_B[self.missing_phase == 1] = -1

        return np.concatenate([pressure_A, running_A, pressure_B, running_B, self.junction_phase_sizes],  axis=1)

    def inside_eval(self):
        state, time = self.eng.get_vehicle_status(), self.eng.get_vehicle_time()
        mask = np.zeros(len(state), bool)
        mask[self.ids] = True

        att = time[(state == 2) & mask].mean()
        tp = ((state == 2) & mask).sum()
        return att, tp

    def step(self, action):
        self.eng.set_tl_phase_batch(self.jids, action)
        self.eng.next_step(self.step_size)
        s = self.observe()
        cnt = self.eng.get_lane_waiting_vehicle_counts()
        if self.reward == 'queue':
            r = cnt[self.in_lane_array]
            r[self.zero_in_lane_array == 1] = 0
            r = -np.sum(r, axis=1)
        if self.reward == 'sum_queue':
            r = [-np.mean([np.sum(cnt[phase_lanes[0]]) for phase_lanes in self.phase_lanes[i]]) for i in range(len(self.jids))]
        if self.reward == 'one_hop_queue':
            r = np.array([-np.sum(cnt[i]) for i in self.in_lanes])
            r_neighbour = np.dot(r, self.connect_matrix)
            r = r + self.alpha*r_neighbour
        if self.reward == 'one_hop_sum_queue':
            r = cnt[self.in_lane_array]
            r[self.zero_in_lane_array == 1] = 0
            r = -np.sum(r, axis=1)
            r = r + self.alpha*np.dot(r, self.connect_matrix)
        if self.reward == 'pressure':
            r = [-np.abs(np.sum(cnt[self.in_lanes[i]])-np.sum(cnt[self.out_lanes[i]])) for i in range(len(self.in_lanes))]
        if self.reward == 'one_hop_pressure':
            r = np.array([-np.abs(np.sum(cnt[self.in_lanes[i]])-np.sum(cnt[self.out_lanes[i]])) for i in range(len(self.in_lanes))])
            r_neighbour = np.dot(r, self.connect_matrix)
            r = r + self.alpha*r_neighbour
        self.info['reward'] = np.mean(r)

        queue_length = cnt[self.in_lane_array]
        queue_length[self.zero_in_lane_array == 1] = 0
        queue_length = np.sum(queue_length, axis=1)
        self.info['AQL'] = np.mean(queue_length)

        self._step += 1
        done = False
        if self._step >= self.step_count:
            self.info['ATT'] = self.eng.get_departed_vehicle_average_traveling_time()
            self.info['ATT_finished'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            self._step = 0
            self.reset()
            done = True
            with open(f'{self.log_dir}/info.log', 'a') as f:
                f.write(f"{self.info['ATT']:.3f} {self.info['Throughput']} {time.time():.3f}\n")
        return s, r, done, self.info


class Replay:
    def __init__(self, max_size):
        self._data = deque([], max_size)

    def len(self):
        return len(self._data)

    def add(self, s, a, r, sp, d, ac, ns, nm, nns, nnm):
        self._data.append([s, a, r, sp, d, ac, ns, nm, nns, nnm])

    def sample(self, batchsize, transpose=False):
        s, a, r, sp, d, ac, ns, nm, nns, nnm = zip(*random.sample(self._data, min(len(self._data), batchsize)))
        if transpose:
            s, a, r, sp, ac, ns, nm, nns, nnm = (list(zip(*i)) for i in [s, a, r, sp, ac, ns, nm, nns, nnm])
        return s, a, r, sp, d, ac, ns, nm, nns, nnm


def make_mlp(*sizes, act=nn.GELU, dropout=0.1):
    layers = []
    for i, o in zip(sizes, sizes[1:]):
        layers.append(nn.Linear(i, o))
        layers.append(act())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers[:-2])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def lerp(a, b, t):
    t = min(1, t)
    return a*(1-t)+b*t


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, mlp_layer):
        super().__init__()
        self.proj = make_mlp(obs_size, *(int(i) for i in mlp_layer.split(',')), action_size)

    def forward(self, obs, neighbor_obs=None, neighbor_mask=None):
        return self.proj(obs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument('--data', type=str, default='./data/us_newyork')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--training_step', type=int, default=10000000)
    parser.add_argument('--training_start', type=int, default=2000)
    parser.add_argument('--training_freq', type=int, default=10)
    parser.add_argument('--target_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--mlp', type=str, default='256,256')
    parser.add_argument('--explore_steps', type=int, default=100000)
    parser.add_argument("--alpha", type=float, default=0.2, help='balance of neighbour rewards')
    parser.add_argument("--mini_batch_size", type=int, default=1024)

    args = parser.parse_args()

    if args.exp is None:
        path = time.strftime('log/advanced_mplight/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/advanced_mplight/{args.exp}/%Y%m%d-%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    with open(f'{path}/cmd.sh', 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\ntensorboard --port 8888 --logdir '+os.path.abspath(path))
    with open(f'{path}/args.json', 'w') as f:
        json.dump(vars(args), f)
    print('tensorboard --port 8888 --logdir '+os.path.abspath(path))

    writer = SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda")

    env = Env(
        data_path=args.data,
        start_step=args.start,
        step_size=args.interval,
        step_count=args.steps//args.interval,
        log_dir=path,
        reward='pressure',
        alpha=args.alpha,
    )
    args.num_agents = len(env.jids)
    # print(env.action_sizes)
    # Q-value 函数，纬度为 [#obs, #action]

    obs = env.observe()
    action_one_hot = np.zeros((args.num_agents, env.max_action_size))
    for i, j in enumerate(env.action_sizes):
        action_one_hot[i, 0] = 1
    # action改成one-hot的
    obs = np.concatenate([obs, action_one_hot], axis=1)
    max_neighbor_num = env.max_neighbor_num
    neighbor_idxs = env.junction_src_list
    obs_sizes = obs.shape[1]
    neighbor_obs = np.zeros((args.num_agents, max_neighbor_num, obs_sizes))
    for i, idxs in enumerate(neighbor_idxs):
        neighbor_obs[i, :len(idxs), :] = obs[idxs, :]

    neighbor_mask = np.zeros((args.num_agents, max_neighbor_num))
    for i, idxs in enumerate(neighbor_idxs):
        neighbor_mask[i, :len(idxs)] = 1

    print(f'{args.num_agents} agents:')
    Q = Actor(obs_sizes, env.max_action_size, args.mlp).to(device)
    opt = optim.AdamW(Q.parameters(), lr=args.lr)
    Q_target = deepcopy(Q)
    replay = Replay(args.buffer_size)
    episode_reward = 0
    episode_step = 0
    available_actions = env.available_actions
    best_episode_reward = -1e999
    aql = 0
    episode_count = args.steps//args.interval

    basic_batch_size = args.batchsize
    basic_update_times = 5
    replay_max = args.buffer_size
    with tqdm(range(args.training_step), ncols=100, smoothing=0.1) as bar:
        for step in bar:
            _st = time.time()
            eps = lerp(1, 0.05, step/args.explore_steps)  # 之前是lerp(1, 0.05, step/100000)
            action_explore = [random.randint(0, a-1) for a in env.action_sizes]
            if step <= args.training_start:
                action = action_explore
            else:
                with torch.no_grad():
                    m = Q(torch.tensor(obs, dtype=torch.float32, device=device), torch.tensor(neighbor_obs, dtype=torch.float32, device=device), torch.tensor(neighbor_mask, dtype=torch.float32, device=device))
                    m[torch.tensor(available_actions, dtype=torch.float32, device=device) == 0] = -1e9
                    action_exploit = torch.argmax(m, dim=-1).cpu().numpy()
                action = np.choose(np.random.uniform(size=args.num_agents) < eps, [action_explore, action_exploit])
            action_one_hot = np.zeros((args.num_agents, env.max_action_size))
            for i, j in enumerate(env.action_sizes):
                action_one_hot[i, action[i]] = 1
            next_obs, reward, done, info = env.step(action)
            next_obs = np.concatenate([next_obs, action_one_hot], axis=1)
            next_neighbor_obs = np.zeros((args.num_agents, max_neighbor_num, obs_sizes))
            for i, idxs in enumerate(neighbor_idxs):
                next_neighbor_obs[i, :len(idxs), :] = next_obs[idxs, :]
            episode_reward += info['reward']
            aql += info['AQL']
            if done:
                writer.add_scalar('metric/EpisodeReward', episode_reward/(args.steps//args.interval), episode_step)
                if episode_reward / episode_count > best_episode_reward:
                    best_episode_reward = episode_reward / episode_count
                    best_aql = aql / episode_count
                    writer.add_scalar('metric/Best_EpisodeReward', episode_reward / episode_count)
                    writer.add_scalar('metric/Best_ATT', info['ATT'])
                    writer.add_scalar('metric/Best_Throughput', info['Throughput'])
                    writer.add_scalar('metric/Best_AQL', best_aql)
                episode_step += 1
                episode_reward = 0
                writer.add_scalar('metric/ATT', info['ATT'], step)
                writer.add_scalar('metric/Throughput', info['Throughput'], step)
                writer.add_scalar('metric/AQL', info['AQL'], step)
                writer.add_scalar('metric/ATT_inside', info['ATT_inside'], step)
                writer.add_scalar('metric/Throughput_inside', info['Throughput_inside'], step)
                aql = 0
            writer.add_scalar('metric/Reward', info['reward'], step)
            replay.add(obs, action, reward, next_obs, done, available_actions, neighbor_obs, neighbor_mask, next_neighbor_obs, neighbor_mask)
            obs = next_obs
            neighbor_obs = next_neighbor_obs
            if step >= args.training_start and step % args.training_freq == 0:
                replay_len = replay.len()
                k = 1 + replay_len / replay_max

                batch_size = int(k * basic_batch_size)
                update_times = int(k * basic_update_times)

                for _ in range(update_times):
                    s, a, r, sp, d, ac, ns, nm, nns, nnm = replay.sample(batch_size, transpose=True)
                    d = torch.tensor(d, dtype=torch.float32, device=device)

                    mini_batch_size = args.mini_batch_size
                    for i in range(int(np.ceil(args.batchsize/mini_batch_size))):
                        s_tmp, a_tmp, r_tmp, sp_tmp, ac_tmp = np.array(s[i*mini_batch_size:(i+1)*mini_batch_size]), a[i*mini_batch_size:(i+1)*mini_batch_size], \
                            r[i*mini_batch_size:(i+1)*mini_batch_size], np.array(sp[i*mini_batch_size:(i+1)*mini_batch_size]), np.array(ac[i*mini_batch_size:(i+1)*mini_batch_size])
                        d_tmp = d.repeat(s_tmp.shape[0], 1).reshape(-1, 1).squeeze(1)
                        s_tmp = torch.tensor(s_tmp, dtype=torch.float32, device=device).reshape(-1, s_tmp.shape[-1]).squeeze(1)
                        a_tmp = torch.tensor(a_tmp, dtype=torch.long, device=device).reshape(-1, 1).squeeze(1)
                        r_tmp = torch.tensor(r_tmp, dtype=torch.float32, device=device).reshape(-1, 1).squeeze(1)
                        sp_tmp = torch.tensor(sp_tmp, dtype=torch.float32, device=device).reshape(-1, sp_tmp.shape[-1]).squeeze(1)
                        ac_tmp = torch.tensor(ac_tmp, dtype=torch.float32, device=device).reshape(-1, ac_tmp.shape[-1]).squeeze(1)
                        with torch.no_grad():
                            m = Q_target(sp_tmp)
                            m[ac_tmp == 0] = -1e9
                            y_target = r_tmp+args.gamma*m.max(1).values*(1-d_tmp)
                        y = Q(s_tmp).gather(-1, a_tmp[..., None]).view(-1)
                        loss = F.mse_loss(y, y_target)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                writer.add_scalar('chart/loss', loss.item(), step)
                bar.set_description(f'ATT: {info["ATT"]:.3f} TP: {info["Throughput"]} ')
                if step % args.target_freq == 0:
                    Q_target.load_state_dict(Q.state_dict())
            writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
    writer.close()


if __name__ == '__main__':
    main()
