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
from run_advanced_mplight import Env as EnvBase
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out


class Env(EnvBase):
    def observe(self):
        cnt = self.eng.get_lane_waiting_vehicle_counts()
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

        return np.concatenate([pressure_A, pressure_B, self.junction_phase_sizes],  axis=1)


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
        path = time.strftime('log/efficient_mplight/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/efficient_mplight/{args.exp}/%Y%m%d-%H%M%S')
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
    basic_update_times = 1
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
