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
from utils.cityflow_engine import get_cityflow_engine
from utils.moss_engine import get_moss_engine
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter # type:ignore
from tqdm import tqdm
from utils.invariant import Colight


def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out

class EnvBase:
    def __init__(self, data_path, start_step, step_size, step_count, log_dir, reward, engine_type ,alpha=0):
        self.log_dir = log_dir
        self.engine_type = engine_type
        if engine_type=="moss":
            self.eng = get_moss_engine(
                    map_file=f'{data_path}/map.bin',
                    agent_file=f'{data_path}/agents.bin',
                    start_step=start_step,
                ) 
        elif engine_type=="cityflow":
            self.eng = get_cityflow_engine(
            config_file=f'{data_path}',
            offset_time=float(start_step)
        ) 
        else:
            raise ValueError(f"Unsupported engine type {engine_type}")

        self.alpha = alpha
        self.jids = [i for i, j in enumerate(self.eng.get_junction_phase_counts()) if j > 1]
        _length_dict = self.eng.get_lane_length_dict() # lid->lane_length
        self.in_lanes, self.out_lanes, self.jpl, self.jpl_label = self.eng.colight_lanes_collect(self.jids)

        def in_lane_numpy(in_lanes):
            max_in_lane_num = max([len(i) for i in in_lanes])
            in_lanes_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            in_lanes_zero_array = np.zeros((len(in_lanes), max_in_lane_num), dtype=int)
            for i, in_lane in enumerate(in_lanes):
                in_lanes_array[i, :len(in_lane)] = in_lane
                in_lanes_zero_array[i, len(in_lane):] = 1
            return in_lanes_array, in_lanes_zero_array

        def phase_lane_numpy(phase_lanes):
            max_inflow_lane_num = max_outflow_lane_num = max(max([max([len(j[1]) for j in i]) for i in phase_lanes]),  max([max([len(j[0]) for j in i]) for i in phase_lanes]))
            phase_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            phase_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            zero_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            zero_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            non_zeros_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            non_zeros_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            missing_lanes_inflow = np.zeros((len(phase_lanes), self.max_action_size, max_inflow_lane_num), dtype=int)
            missing_lanes_outflow = np.zeros((len(phase_lanes), self.max_action_size, max_outflow_lane_num), dtype=int)
            missing_phase = np.zeros((len(phase_lanes), self.max_action_size), dtype=int)

            for i, phase_lane in enumerate(phase_lanes):
                for j, lanes in enumerate(phase_lane):
                    phase_lanes_inflow[i, j, :len(lanes[0])] = lanes[0]
                    phase_lanes_outflow[i, j, :len(lanes[1])] = lanes[1]
                    zero_lanes_inflow[i, j, len(lanes[0]):] = 1
                    zero_lanes_outflow[i, j, len(lanes[1]):] = 1
                    non_zeros_inflow[i, j, :len(lanes[0])] = 1
                    non_zeros_outflow[i, j, :len(lanes[1])] = 1
                if len(phase_lane) < self.max_action_size:
                    for j in range(len(phase_lane), self.max_action_size):
                        missing_lanes_inflow[i, j, :] = 1
                        missing_lanes_outflow[i, j, :] = 1
                        non_zeros_inflow[i, j, :] = 1
                        non_zeros_outflow[i, j, :] = 1
                        missing_phase[i, j] = 1
            return phase_lanes_inflow, phase_lanes_outflow, zero_lanes_inflow, zero_lanes_outflow, missing_lanes_inflow, missing_lanes_outflow, non_zeros_inflow, non_zeros_outflow, missing_phase

        self.action_sizes = list(self.eng.get_junction_phase_counts())
        self.action_sizes = [self.action_sizes[i] for i in self.jids]
        self.max_action_size = max(self.action_sizes)
        self.available_actions = np.zeros((len(self.action_sizes), max(self.action_sizes)))
        for i, j in enumerate(self.action_sizes):
            self.available_actions[i, :j] = 1
        self.phase_lanes = self.jpl
        self.junction_type_list = [np.array([(m == 3) | (m == 6), (m == 4) | (m == 8)]).astype(int) for m in self.action_sizes]
        self.junction_scale_list = np.array([[len(i)/j] for i, j in zip(self.in_lanes, self.action_sizes)])

        self.phase_lanes_inflow, self.phase_lanes_outflow, self.zero_lanes_inflow, self.zero_lanes_outflow, self.missing_lanes_inflow, self.missing_lanes_outflow, self.non_zeros_inflow, self.non_zeros_outflow, self.missing_phases = phase_lane_numpy(
            self.phase_lanes)
        # print("self.phase_lanes_inflow",self.phase_lanes_inflow)
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
                        self.junction_graph_edge_feats.append([_length_dict[out_lane], 1])
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

        print(f'Training on {self.engine_type} {len(self.jids)} junctions')
        self._cid = self.eng.make_checkpoint()
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
            "SIM_TIME":0,
            "RL_TIME":0,
        }
        self.data_path = data_path
        self.start_step = start_step
        self.one_hot_mapping_matrix = np.eye(self.max_action_size)

    def reset(self):
        self.eng.reset(self._cid)

    def observe(self):
        cnt = self.eng.get_lane_waiting_vehicle_counts()
        in_cnt_states, out_cnt_states = cnt[self.phase_lanes_inflow], cnt[self.phase_lanes_outflow]
        in_cnt_states[self.zero_lanes_inflow == 1] = 0
        out_cnt_states[self.zero_lanes_outflow == 1] = 0
        in_cnt_states[self.missing_lanes_inflow == 1] = -1
        out_cnt_states[self.missing_lanes_outflow == 1] = -1
        m = np.sum(in_cnt_states, axis=2)  # 相位车道总等待车辆数
        m[self.missing_phases == 1] = -1

        def div(a, b, fill=0):
            mask = b == 0
            x = a/(b+1e-6)
            x[mask] = fill
            return x
        n = div(  # 相位车道平均等待车辆数
            np.sum(in_cnt_states, axis=2),
            np.sum(self.non_zeros_inflow, axis=2),
            -1
        )
        observe_states = np.concatenate([m, n, self.junction_phase_sizes], axis=1)
        return observe_states

    def step(self, action):
        _step_start_time = time.time()
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
        self.info['reward'] = np.mean(r) # type:ignore

        queue_length = cnt[self.in_lane_array]
        queue_length[self.zero_in_lane_array == 1] = 0
        queue_length = np.sum(queue_length, axis=1)
        self.info['AQL'] = np.mean(queue_length)

        self._step += 1
        done = False
        _step_end_time = time.time()
        self.info["SIM_TIME"]+=(_step_end_time-_step_start_time)
        if self._step >= self.step_count:
            self.info['ATT'] = self.eng.get_departed_vehicle_average_traveling_time()
            self.info['ATT_finished'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            self._step = 0
            _rl_time = self.info["RL_TIME"]
            _sim_time = self.info["SIM_TIME"]
            self.info["RL_TIME"] = 0
            self.info["SIM_TIME"] = 0
            self.reset()
            done = True
            with open(f'{self.log_dir}/info.log', 'a') as f:
                _iter_stop_time = time.time()
                f.write(f"{self.info['ATT']:.3f} {self.info['Throughput']} {_iter_stop_time:.3f} {_sim_time:3f} {_rl_time:3f}\n")
        return s, r, done, self.info # type:ignore

class Env(EnvBase):
    def observe(self):
        cnt = self.eng.get_lane_waiting_vehicle_counts()
        in_cnt_states, out_cnt_states = cnt[self.phase_lanes_inflow], cnt[self.phase_lanes_outflow]
        in_cnt_states[self.zero_lanes_inflow == 1] = 0
        out_cnt_states[self.zero_lanes_outflow == 1] = 0
        in_cnt_states[self.missing_lanes_inflow == 1] = -1
        out_cnt_states[self.missing_lanes_outflow == 1] = -1
        m = np.sum(in_cnt_states, axis=2)-np.sum(out_cnt_states, axis=2)
        m[self.missing_phases == 1] = -1

        def div(a, b, fill=0):
            mask = b == 0
            x = a/(b+1e-6)
            x[mask] = fill
            return x
        n = div(
            np.sum(in_cnt_states, axis=2),
            np.sum(self.non_zeros_inflow, axis=2),
            -1
        )
        observe_states = np.concatenate([m, n, self.junction_phase_sizes], axis=1)
        return observe_states


class Replay:
    def __init__(self, max_size):
        self._data = deque([], max_size)

    def len(self):
        return len(self._data)

    def add(self, s, a, r, sp, d, ns, nm, nns, nnm):
        self._data.append([s, a, r, sp, d, ns, nm, nns, nnm])

    def sample(self, batchsize, transpose=False):
        s, a, r, sp, d, ns, nm, nns, nnm = zip(*random.sample(self._data, min(len(self._data), batchsize)))
        if transpose:
            s, a, r, sp, ns, nm, nns, nnm = (list(zip(*i)) for i in [s, a, r, sp, ns, nm, nns, nnm])
        return s, a, r, sp, d, ns, nm, nns, nnm


def make_mlp(*sizes, act=nn.GELU, dropout=0.1):
    layers = []
    for i, o in zip(sizes, sizes[1:]):
        layers.append(nn.Linear(i, o))
        layers.append(act())
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers[:-2])


class DQN(nn.Module):
    def __init__(self, obs_size, action_size, mlp='256,256'):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        hidden_dim = 32
        self.invariant = Colight(obs_size, 'colight', hidden_dim=hidden_dim, heads=5)
        self.Q = make_mlp(obs_size+hidden_dim, *(int(i) for i in mlp.split(',')), action_size)

    def forward(self, x, neighbor_obs=None, neighbor_masks=None):
        obs_n = self.invariant(x, neighbor_obs, neighbor_masks)
        obs = torch.cat([x, obs_n], dim=-1)
        return self.Q(obs)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def lerp(a, b, t):
    t = min(1, t)
    return a*(1-t)+b*t


def load(models, save_dir, type='best'):
    for idx, model in enumerate(models):
        model.load_state_dict(torch.load(str(save_dir) + "/model_{}_{}.pt".format(type, idx)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument('--data', type=str, default='data_cb/Nanchang')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--training_step', type=int, default=10000000)
    parser.add_argument('--training_start', type=int, default=1000)
    parser.add_argument('--training_freq', type=int, default=10)
    parser.add_argument('--target_freq', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--buffer_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--mlp', type=str, default='256,256')
    parser.add_argument('--engine_type', type=str, default='moss')
    args = parser.parse_args()

    if args.exp is None:
        path = time.strftime('log/colight/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/colight/{args.exp}/%Y%m%d-%H%M%S')
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
        engine_type=args.engine_type,
    )

    def cal_neighbor_obs(obs, neighbor_idxs, max_neighbor_num):
        obs = np.array(obs)
        neighbor_obs = np.zeros((obs.shape[0], max_neighbor_num, obs.shape[1]), dtype=np.float32)
        neighbor_mask = np.zeros((obs.shape[0], max_neighbor_num), dtype=np.float32)
        for i, idxs in enumerate(neighbor_idxs):
            neighbor_obs[i, :len(idxs), :] = np.array(obs[idxs, :]).copy()
            neighbor_mask[i, :len(idxs)] = 1
        return neighbor_obs, neighbor_mask

    obs = env.observe()
    neighbor_idxs = env.junction_src_list
    max_neighbor_num = env.max_neighbor_num
    neighbor_obs, neighbor_mask = cal_neighbor_obs(obs, env.junction_src_list, env.max_neighbor_num)
    obs_sizes = [len(i) for i in obs]
    print(f'{len(obs_sizes)} agents:')

    Q = [DQN(o, a, args.mlp).to(device) for o, a in zip(obs_sizes, env.action_sizes)]

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
            for q, o, a, ns, nm in zip(Q, obs, env.action_sizes, neighbor_obs, neighbor_mask):
                ns, nm = np.array(ns), np.array(nm)
                if step < args.training_start or random.random() < eps:  # explore
                    action.append(random.randint(0, a-1))
                else:
                    ns = [torch.tensor(ns[i], dtype=torch.float32, device=device).unsqueeze(0) for i in range(len(ns))]
                    nm = [torch.tensor(nm[i], dtype=torch.float32, device=device).unsqueeze(0) for i in range(len(nm))]
                    action.append(
                        torch.argmax(q(torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0), ns, nm).squeeze(0)))
            next_obs, reward, done, info = env.step(action)
            _rl_start_time = time.time()
            next_neighbor_obs, next_neighbor_mask = cal_neighbor_obs(next_obs, neighbor_idxs, max_neighbor_num)
            episode_reward += info['reward']
            if done:
                writer.add_scalar('metric/EpisodeReward', episode_reward/(args.steps//args.interval), episode_step)
                if episode_reward > best_episode_reward:
                    best_episode_reward = episode_reward
                    writer.add_scalar('metric/Best_EpisodeReward', episode_reward/(args.steps//args.interval))
                    writer.add_scalar('metric/Best_ATT', info['ATT'])
                    writer.add_scalar('metric/Best_Throughput', info['Throughput'])
                episode_step += 1
                episode_reward = 0
                writer.add_scalar('metric/ATT', info['ATT'], step)
                writer.add_scalar('metric/Throughput', info['Throughput'], step)
                writer.add_scalar('metric/ATT_inside', info['ATT_inside'], step)
                writer.add_scalar('metric/Throughput_inside', info['Throughput_inside'], step)
            writer.add_scalar('metric/Reward', info['reward'], step)
            replay.add(obs, action, reward, next_obs, done, neighbor_obs, neighbor_mask, next_neighbor_obs, next_neighbor_mask)
            obs = next_obs
            neighbor_obs, neighbor_mask = next_neighbor_obs, next_neighbor_mask

            if step >= args.training_start and step % args.training_freq == 0:
                replay_len = replay.len()
                # k = 1 + replay_len / replay_max
                k = 1

                batch_size = int(k * basic_batch_size)
                update_times = int(k * basic_update_times)
                for _ in range(update_times):
                    s, a, r, sp, d, ns, nm, nns, nnm = replay.sample(batch_size, transpose=True)
                    d = torch.tensor(d, dtype=torch.float32, device=device)
                    loss = 0
                    for q, qt, s, a, r, sp, ns, nm, nns, nnm in zip(Q, Q_target, s, a, r, sp, ns, nm, nns, nnm):
                        ns, nm, nns, nnm = np.array(ns), np.array(nm), np.array(nns), np.array(nnm)
                        s = torch.tensor(np.array(s), dtype=torch.float32, device=device)
                        a = torch.tensor(a, dtype=torch.long, device=device)
                        r = torch.tensor(r, dtype=torch.float32, device=device)
                        sp = torch.tensor(np.array(sp), dtype=torch.float32, device=device)
                        ns = [torch.tensor(ns[:, i], dtype=torch.float32, device=device) for i in range(max_neighbor_num)]
                        nm = [torch.tensor(nm[:, i], dtype=torch.float32, device=device) for i in range(max_neighbor_num)]
                        nns = [torch.tensor(nns[:, i], dtype=torch.float32, device=device) for i in range(max_neighbor_num)]
                        nnm = [torch.tensor(nnm[:, i], dtype=torch.float32, device=device) for i in range(max_neighbor_num)]
                        with torch.no_grad():
                            y_target = r+args.gamma*qt(sp, nns, nnm).max(1).values*(1-d)
                        y = q(s, ns, nm).gather(-1, a[..., None]).view(-1)
                        loss = loss+F.mse_loss(y, y_target)
                    loss = loss/len(Q)
                    opt.zero_grad()
                    loss.backward() # type:ignore
                    opt.step()
                writer.add_scalar('chart/loss', loss.item(), step) # type:ignore
                bar.set_description(f'ATT: {info["ATT"]:.3f} TP: {info["Throughput"]} ')
                if step % args.target_freq == 0:
                    for a, b in zip(Q, Q_target):
                        b.load_state_dict(a.state_dict())
            writer.add_scalar('chart/FPS', 1/(time.time()-_st), step)
            _rl_end_time = time.time()
            env.info["RL_TIME"]+=(_rl_end_time-_rl_start_time)
    writer.close()


if __name__ == '__main__':
    main()
