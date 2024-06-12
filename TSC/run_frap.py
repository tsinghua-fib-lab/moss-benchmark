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
        in_cnt_states_A = cnt[self.phase_lanes_inflow_A]
        in_cnt_states_A[self.zero_lanes_inflow_A == 1] = 0
        in_cnt_states_B = cnt[self.phase_lanes_inflow_B]
        in_cnt_states_B[self.zero_lanes_inflow_B == 1] = 0

        in_cnt_A = np.sum(in_cnt_states_A, axis=2)
        in_cnt_A[self.missing_phase == 1] = -1
        in_cnt_B = np.sum(in_cnt_states_B, axis=2)
        in_cnt_B[self.missing_phase == 1] = -1
        return in_cnt_A, in_cnt_B

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
        sa, sb = self.observe()
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
        return sa, sb, r, done, self.info


class Replay:
    def __init__(self, max_size):
        self._data = deque([], max_size)

    def len(self):
        return len(self._data)

    def add(self, sa, sb, a, aoh, r, nsa, nsb, naoh, d):
        self._data.append([sa, sb, a, aoh, r, nsa, nsb, naoh, d])

    def sample(self, batchsize, transpose=False):
        sa, sb, a, aoh, r, nsa, nsb, naoh, d = zip(*random.sample(self._data, min(len(self._data), batchsize)))
        if transpose:
            sa, sb, a, aoh, r, nsa, nsb, naoh = (list(zip(*i)) for i in [sa, sb, a, aoh, r, nsa, nsb, naoh])
        return sa, sb, a, aoh, r, nsa, nsb, naoh, d


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

# phase-level actor


class Actor(nn.Module):
    def __init__(self, obs_size, action_size, device):
        super().__init__()
        self.device = device
        self.linear_h1 = nn.Linear(1, 8)
        self.relu_h1 = nn.ReLU()
        self.linear_h2 = nn.Linear(1, 8)
        self.relu_h2 = nn.ReLU()
        self.action_sizes = action_size
        self.linear = nn.Linear(16, 16)
        self.conv1 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, obs_h1_A, obs_h1_B, obs_h2):
        phase_representations = torch.zeros((obs_h1_A.shape[0], self.action_sizes, 16), dtype=torch.float32, device=self.device)
        for phase_idx in range(self.action_sizes):
            obs_1_A = self.relu_h1(self.linear_h1(obs_h1_A[:, [phase_idx]]))
            obs_2_A = self.relu_h1(self.linear_h2(obs_h2[:, [phase_idx]]))
            obs_A = self.linear(torch.cat([obs_1_A, obs_2_A], axis=1))
            obs_1_B = self.relu_h1(self.linear_h1(obs_h1_B[:, [phase_idx]]))
            obs_2_B = self.relu_h1(self.linear_h2(obs_h2[:, [phase_idx]]))
            obs_B = self.linear(torch.cat([obs_1_B, obs_2_B], axis=1))
            phase_representations[:, phase_idx, :] = obs_A + obs_B

        phase_demand_embedding_matrix = torch.zeros((obs_h1_A.shape[0], 32, self.action_sizes, self.action_sizes-1), dtype=torch.float32, device=self.device)
        for phase_idx in range(self.action_sizes):
            count = 0
            for competing_phase in range(self.action_sizes):
                if phase_idx == competing_phase:
                    continue
                phase_demand_embedding_matrix[:, :, phase_idx, count] = torch.cat([phase_representations[:, phase_idx], phase_representations[:, competing_phase]], dim=1)
                count += 1
        phase_demand_embedding_matrix = self.bn1(self.conv1(phase_demand_embedding_matrix))
        phase_demand_embedding_matrix = self.bn2(self.conv2(phase_demand_embedding_matrix))
        phase_demand_embedding_matrix = self.conv3(phase_demand_embedding_matrix).squeeze(1)
        phase_scores = torch.sum(phase_demand_embedding_matrix, axis=2)
        return phase_scores


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
        path = time.strftime('log/frap%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/frap/{args.exp}/%Y%m%d-%H%M%S')
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
        reward='queue',
        alpha=args.alpha,
    )
    args.num_agents = len(env.jids)

    obs_a, obs_b = env.observe()
    action_one_hot = np.zeros((args.num_agents, env.max_action_size))
    for i, j in enumerate(env.action_sizes):
        action_one_hot[i, 0] = 1
    obs_sizes = [len(i) for i in obs_a]
    print(f'{len(obs_sizes)} agents')

    Q = [Actor(o, a, device).to(device) for o, a in zip(obs_sizes, env.action_sizes)]

    opt = optim.AdamW([j for i in Q for j in i.parameters()], lr=args.lr)
    Q_target = deepcopy(Q)
    replay = Replay(args.buffer_size)
    episode_reward = 0
    episode_step = 0
    best_episode_reward = -1e999
    basic_batch_size = args.batchsize
    basic_update_times = 1
    with tqdm(range(args.training_step), ncols=100, smoothing=0.1) as bar:
        for step in bar:
            _st = time.time()
            eps = lerp(1, 0.05, step/100000)
            action = []
            for q, oa, ob, oac, a in zip(Q, obs_a, obs_b, action_one_hot, env.action_sizes):
                if step < args.training_start or random.random() < eps:  # explore
                    action.append(random.randint(0, a-1))
                else:
                    action.append(
                        torch.argmax(q(torch.tensor(oa, dtype=torch.float32, device=device).unsqueeze(0), torch.tensor(ob, dtype=torch.float32, device=device).unsqueeze(0), torch.tensor(oac, dtype=torch.float32, device=device).unsqueeze(0)).squeeze(0)).cpu().numpy())
            next_obs_a, next_obs_b, reward, done, info = env.step(action)
            next_action_one_hot = np.zeros((args.num_agents, env.max_action_size))
            for i, j in enumerate(action):
                next_action_one_hot[i, j] = 1

            episode_reward += info['reward']
            if done:
                writer.add_scalar('metric/EpisodeReward', episode_reward/(args.steps//args.interval), episode_step)
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    writer.add_scalar('metric/Best_EpisodeReward', episode_reward/(args.steps//args.interval))
                    writer.add_scalar('metric/Best_ATT', info['ATT'])
                    writer.add_scalar('meQtric/Best_Throughput', info['Throughput'])
                episode_step += 1
                episode_reward = 0
                writer.add_scalar('metric/ATT', info['ATT'], step)
                writer.add_scalar('metric/Throughput', info['Throughput'], step)
                writer.add_scalar('metric/ATT_inside', info['ATT_inside'], step)
                writer.add_scalar('metric/Throughput_inside', info['Throughput_inside'], step)
            writer.add_scalar('metric/Reward', info['reward'], step)
            replay.add(obs_a, obs_b, action, action_one_hot, reward, next_obs_a, next_obs_b, next_action_one_hot, done)
            obs_a, obs_b = next_obs_a, next_obs_b
            action_one_hot = next_action_one_hot

            if step >= args.training_start and step % args.training_freq == 0:
                k = 1

                batch_size = int(k * basic_batch_size)
                update_times = int(k * basic_update_times)
                for _ in range(update_times):
                    sa, sb, a, aoh, r, spa, spb, saoh, d = replay.sample(batch_size, transpose=True)
                    d = torch.tensor(d, dtype=torch.float32, device=device)
                    loss = 0
                    for q, qt, sa, sb, a, aoh, r, spa, spb, saoh in zip(Q, Q_target, sa, sb, a, aoh, r, spa, spb, saoh):
                        sa = torch.tensor(np.array(sa), dtype=torch.float32, device=device)
                        sb = torch.tensor(np.array(sb), dtype=torch.float32, device=device)
                        a = torch.tensor(np.array(a), dtype=torch.long, device=device)
                        aoh = torch.tensor(np.array(aoh), dtype=torch.float32, device=device)
                        r = torch.tensor(np.array(r), dtype=torch.float32, device=device)
                        spa = torch.tensor(np.array(spa), dtype=torch.float32, device=device)
                        spb = torch.tensor(np.array(spb), dtype=torch.float32, device=device)
                        saoh = torch.tensor(np.array(saoh), dtype=torch.float32, device=device)
                        with torch.no_grad():
                            y_target = r+args.gamma*qt(spa, spb, saoh).max(1).values*(1-d)
                        y = q(sa, sb, aoh).gather(-1, a[..., None]).view(-1)
                        loss = loss+F.mse_loss(y, y_target)
                    loss = loss/len(Q)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                writer.add_scalar('chart/loss', loss.item(), step)
                bar.set_description(f'ATT: {info["ATT"]:.3f} TP: {info["Throughput"]} ')
                if step % args.target_freq == 0:
                    for a, b in zip(Q, Q_target):
                        b.load_state_dict(a.state_dict())
            writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
    writer.close()


if __name__ == '__main__':
    main()
