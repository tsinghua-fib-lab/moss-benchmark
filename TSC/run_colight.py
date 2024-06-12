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
from run_mplight import Env as EnvBase
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.invariant import Colight


def decompose_action(x, sizes):
    out = []
    for i in sizes:
        x, r = divmod(x, i)
        out.append(r)
    return out


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
    parser.add_argument('--data', type=str, default='./data/us_newyork')
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
