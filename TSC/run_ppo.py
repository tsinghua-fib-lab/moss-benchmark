import argparse
import json
import os
import random
import shutil
import sys
import time
from glob import glob

import numpy as np
import torch
from engine import get_engine
from torch import nn, optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

NN_INPUT_SCALER = 5


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument("--suffix", type=str)
    parser.add_argument("--seed", type=int, default=43, help="seed of the experiment")

    parser.add_argument('--data', type=str, default='./data/us_newyork')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=3600)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--mlp', type=str, default='256,256')

    parser.add_argument("--total-timesteps", type=int, default=100000000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-scale", type=float, default=0)
    parser.add_argument("--lr-anneal", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae", type=bool, default=True, help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.9, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=8, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=16, help="the K epochs to update the policy")
    parser.add_argument("--ent-coef", type=float, default=0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--clip-coef", type=float, default=0.25, help="the surrogate clipping coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument('--norm-adv', action='store_true')

    parser.add_argument("--load", type=str, default="", help='model checkpoint')
    parser.add_argument('--load-step', type=int, default=-1)
    parser.add_argument('--num-steps', type=int, default=128, help='env rollout steps')
    parser.add_argument('--save-interval', type=int, default=1000, help='checkpoint save interval')

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--sample', action='store_true')

    args = parser.parse_args()
    args.debug |= args.eval
    args.batch_size = args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    return args


def pad(arr, value):
    l = max(len(i) for i in arr)
    return np.array([
        i+[value]*(l-len(i)) for i in arr
    ])


class Env:
    def __init__(self, data_path, start_step, step_size, step_count, log_dir, max_veh_cnt=200):
        self.log_dir = log_dir
        self.max_veh_cnt = max_veh_cnt
        self.eng = eng = get_engine(
            map_file=f'{data_path}/map.bin',
            agent_file=f'{data_path}/agents.bin',
            start_step=start_step,
        )
        self.j_ids = j_ids = [i for i, j in enumerate(eng.get_junction_phase_counts()) if j > 1]
        l = eng.get_junction_inout_lanes()[0]
        self.in_lanes = [l[i] for i in j_ids]
        self.lane_lengths = np.maximum(1, eng.get_lane_lengths())

        # 计算观测车道
        jpl = eng.get_junction_phase_lanes()
        jpl = [jpl[j] for j in j_ids]
        l_ids = {l for i in jpl for j in i for k in j for l in k}
        self.l_ids = l_ids = sorted(l_ids)
        l_map = {j: i for i, j in enumerate(l_ids)}
        self.p2l_in = pad(
            [[l_map[i] for i in p[0]] for j in jpl for p in j],
            -1
        )
        self.p2l_out = pad(
            [[l_map[i] for i in p[1]] for j in jpl for p in j],
            -1
        )
        j2p = []
        cnt = 0
        for i in jpl:
            j2p.append(list(range(cnt, cnt+len(i))))
            cnt += len(i)
        self.j2p = pad(j2p, -1)
        self.j2l_in = pad(
            [sorted(l_map[l] for j in i for l in j[0]) for i in jpl],
            -1
        )
        self.j2l_out = pad(
            [sorted(l_map[l] for j in i for l in j[1]) for i in jpl],
            -1
        )
        self.obs_size = (len(l_ids), 4)
        self.num_envs = len(j_ids)

        print(f'{len(self.j_ids)} junctions in total')
        self._cid = self.eng.make_checkpoint()
        self.step_size = step_size
        self.step_count = step_count
        self._step = 0
        self.info = {
            'ATT-d': 1e999,
            'ATT-f': 1e999,
            'Throughput': 0,
            'reward': 0,
        }

    def _clip_veh_cnt(self, cnt):
        return np.minimum(self.max_veh_cnt, cnt)/self.max_veh_cnt*NN_INPUT_SCALER

    def reset(self):
        self.eng.restore_checkpoint(self._cid)

    def observe(self):
        c1 = self.eng.get_lane_vehicle_counts()
        c2 = self.eng.get_lane_waiting_at_end_vehicle_counts(distance_to_end=150)
        obs = np.stack([
            self._clip_veh_cnt(c1),
            self._clip_veh_cnt(c2),
            np.clip(c1/self.lane_lengths*3, 0, 1)*NN_INPUT_SCALER,
            np.clip(c2/self.lane_lengths*3, 0, 1)*NN_INPUT_SCALER,
            # np.zeros_like(c1)+self.eng.get_current_time()/(self.step_size*self.step_count)*NN_INPUT_SCALER,
        ]).T
        obs = obs[self.l_ids]
        return obs

    def step(self, action):
        self.eng.set_tl_phase_batch(self.j_ids, action)
        self.eng.next_step(self.step_size)
        s = self.observe()
        cnt = self._clip_veh_cnt(self.eng.get_lane_waiting_at_end_vehicle_counts())
        r = np.array([-np.mean(cnt[i]) for i in self.in_lanes])
        self.info['reward'] = np.mean(r)
        self._step += 1
        done = False
        if self._step >= self.step_count:
            self.info['ATT-d'] = self.eng.get_departed_vehicle_average_traveling_time()
            self.info['ATT-f'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            self._step = 0
            self.reset()
            done = True
            s = self.observe()
            with open(f'{self.log_dir}/info.log', 'a') as f:
                f.write(f"{self.info['ATT-d']:.3f} {self.info['Throughput']} {time.time():.3f}\n")
        return s, r, done, self.info


def make_mlp(*shape, dropout=0.1, act=nn.Tanh, sigma=None, compile=True):
    ls = [nn.Linear(i, j) for i, j in zip(shape, shape[1:])]
    if sigma is not None:
        for l in ls:
            nn.init.orthogonal_(l.weight, 2**0.5)
            nn.init.constant_(l.bias, 0)
        nn.init.orthogonal_(ls[-1].weight, sigma)
    mlp = nn.Sequential(
        *sum(([
            l,
            act(),
            nn.Dropout(dropout),
        ] for l in ls[:-1]), []),
        ls[-1]
    )
    if compile:
        return torch.compile(mlp, fullgraph=True)
    return mlp


def pad_tensor(x, value=0, dim=-1):
    shape = list(x.shape)
    shape[dim] = 1
    return torch.cat([x, value+torch.zeros(*shape, dtype=x.dtype, device=x.device)], dim)


class Model(nn.Module):
    def __init__(self, env: Env, dim_mlp):
        super().__init__()
        self.critic = make_mlp(env.obs_size[-1], *dim_mlp, 1, sigma=1)
        self.actor = make_mlp(env.obs_size[-1],  *dim_mlp, 1, sigma=0.01)
        self.critic_alpha = nn.Parameter(torch.zeros(1))
        self.register_buffer('p2l_in', torch.LongTensor(env.p2l_in), False)
        self.register_buffer('p2l_out', torch.LongTensor(env.p2l_out), False)
        self.register_buffer('j2p', torch.LongTensor(env.j2p), False)
        self.register_buffer('j2l_in', torch.LongTensor(env.j2l_in), False)
        self.register_buffer('j2l_out', torch.LongTensor(env.j2l_out), False)

    def get_value(self, x):
        # B x L
        l_value = pad_tensor(self.critic(x).squeeze(-1))
        # B x J
        j_value = l_value[:, self.j2l_in].mean(2)-self.critic_alpha*l_value[:, self.j2l_out].mean(2)
        return j_value

    def get_action_and_value(self, x, action=None, sample=True, action_only=False):
        # B x L
        l_value = pad_tensor(
            self.actor(x).squeeze(-1)
        )
        # B x P
        p_value = pad_tensor(
            l_value[:, self.p2l_in].mean(2)-l_value[:, self.p2l_out].mean(2),
            value=-1e999
        )
        # B x J x A
        j_value = p_value[:, self.j2p]
        probs = Categorical(logits=j_value)
        if action is None:
            action = probs.sample() if sample else torch.argmax(j_value, -1)  # B x J
        if action_only:
            return action
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Dummy():
    def add_text(*_):
        pass

    def add_scalar(*_):
        pass

    def add_scalars(*_):
        pass

    def close(*_):
        pass


def make_object(d: dict):
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    return Obj(d)


def main():
    args = parse_args()
    if args.exp is None:
        path = time.strftime('log/ppo/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/ppo/{args.exp}/%Y%m%d-%H%M%S')
    if args.suffix is not None:
        path += '_'+args.suffix
    print('tensorboard --port 8888 --logdir '+os.path.abspath(path))
    if args.load:
        pts = sorted([int(i.rsplit('/', 1)[1][:-3]), i] for i in glob(f'{args.load}/ckpts/*.pt'))
        if args.load_step == -1:
            args.load_step = pts[-1][0]
        pt = min(pts, key=lambda x: abs(x[0]-args.load_step))[1]
        print(f'Load checkpoint from {pt}')
        cfg = json.load(open(f'{args.load}/args.json'))
        assert cfg['mlp'] == args.mlp
    else:
        pt = None
    args.loaded_checkpoint = pt
    if not args.debug:
        os.makedirs(path+'/code')
        os.makedirs(path+'/ckpts')
        shutil.copy(__file__, path+'/code/'+os.path.split(__file__)[1])
        if pt is not None:
            shutil.copy(pt, f'{path}/load.pt')
        with open(f'{path}/cmd.sh', 'w') as f:
            f.write('python ')
            f.write(' '.join(sys.argv))
            f.write('\ntensorboard --port 8888 --logdir '+os.path.abspath(path))
        with open(f'{path}/args.json', 'w') as f:
            json.dump(vars(args), f)

    writer = Dummy() if args.debug else SummaryWriter(path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda')

    env = Env(
        data_path=args.data,
        start_step=args.start,
        step_size=args.interval,
        step_count=args.steps//args.interval,
        log_dir=path,
    )
    dim_mlp = [int(i) for i in args.mlp.split(',')]
    agent = Model(env, dim_mlp).to(device)
    if pt is not None:
        agent.load_state_dict(torch.load(pt, map_location=device))
    if args.eval:
        agent.eval()
        with torch.no_grad():
            next_obs = torch.Tensor(env.observe()).to(device)
            for _ in tqdm(range(env.step_count), ncols=90):
                action = agent.get_action_and_value(next_obs.unsqueeze(0), sample=args.sample, action_only=True)
                next_obs, reward, done, info = env.step(action.view(-1).cpu().numpy())
                next_obs = torch.Tensor(next_obs).to(device)
            print(f'{info["Throughput"]} {info["ATT-d"]:.1f} {info["ATT-f"]:.1f}')
        return
    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    # 每个路口视为1个env，但obs和done是共享的
    obs = torch.zeros((args.num_steps, *env.obs_size), device=device)
    actions = torch.zeros((args.num_steps, env.num_envs), dtype=torch.long, device=device)
    log_probs = torch.zeros((args.num_steps, env.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, env.num_envs), device=device)
    dones = torch.zeros((args.num_steps, env.num_envs), device=device)
    values = torch.zeros((args.num_steps, env.num_envs), device=device)

    global_step = 0
    next_obs = torch.Tensor(env.observe()).to(device)
    next_done = torch.zeros(1, device=device)
    if args.save_interval:
        next_save_step = args.save_interval
    else:
        next_save_step = 1e999
    with tqdm(range(args.total_timesteps), ncols=90, smoothing=0.1) as bar:
        while global_step < args.total_timesteps:
            _t = time.time()
            for step in range(args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                    values[step] = value
                actions[step] = action
                log_probs[step] = log_prob

                next_obs, reward, done, info = env.step(action.view(-1).cpu().numpy())
                rewards[step] = torch.tensor(reward).to(device)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor([done]).to(device)

                global_step += 1
                bar.update(1)
                if done:
                    writer.add_scalar('metric/ATT-d', info['ATT-d'], global_step)
                    writer.add_scalar('metric/ATT-f', info['ATT-f'], global_step)
                    writer.add_scalar('metric/Throughput', info['Throughput'], global_step)
                writer.add_scalar('metric/Reward', info['reward'], global_step)
            writer.add_scalar("charts/Sample Time", time.time()-_t, global_step)

            if os.path.exists(path+'/lr.txt'):
                try:
                    with open(path+'/lr.txt') as f:
                        lr = float(f.read())
                    if lr != optimizer.param_groups[0]["lr"]:
                        print(f'Change lr to {lr}')
                        optimizer.param_groups[0]["lr"] = lr
                        writer.add_scalar("charts/lr", lr, global_step)
                except:
                    pass
            else:
                if args.lr_anneal > 0:
                    optimizer.param_groups[0]["lr"] = lr = args.lr*max(0.1, 1-global_step/args.lr_anneal)
                    writer.add_scalar("charts/lr", lr, global_step)
                elif args.lr_scale > 0:
                    optimizer.param_groups[0]["lr"] = lr = args.lr*min(1, abs(info['reward'])/args.lr_scale)
                    writer.add_scalar("charts/lr", lr, global_step)

            _t = time.time()
            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs.unsqueeze(0))
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    last_gae_lam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1.0 - next_done
                            next_values = next_value
                        else:
                            next_nonterminal = 1.0 - dones[t + 1]
                            next_values = values[t + 1]
                        delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                        advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            next_nonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * next_nonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = obs
            b_log_probs = log_probs
            b_actions = actions
            b_advantages = advantages
            b_returns = returns
            b_values = values

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = np.arange(args.batch_size)
                np.random.shuffle(b_inds)
                b_inds = b_inds.reshape(-1, args.minibatch_size)
                for mb_inds in b_inds:
                    _, new_log_prob, entropy, new_value = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    log_ratio = new_log_prob - b_log_probs[mb_inds]
                    if np.any(log_ratio.detach().cpu().numpy() >= 80):
                        print('Warning: log_ratio too big', log_ratio)
                        log_ratio = torch.clamp(log_ratio, None, 80)
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            writer.add_scalar("charts/Optimize Time", time.time()-_t, global_step)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            ratio = np.array([i.item() for i in [pg_loss, entropy_loss * args.ent_coef, v_loss * args.vf_coef]])
            ratio = ratio / ratio.sum()
            writer.add_scalars("losses/ratio", {
                'policy': ratio[0],
                'entropy': ratio[1],
                'value': ratio[2],
            }, global_step)
            msg = f'{loss.item():.3f} ATT: {info["ATT-d"]:.1f} TP: {info["Throughput"]}'
            bar.set_description(msg)
            if not args.debug:
                with open(f'{path}/msg.log', 'w') as f:
                    f.write(msg)
            # if global_step >= next_save_step:
            #     torch.save(agent.state_dict(), f'{path}/ckpts/{global_step}.pt')
    writer.close()


if __name__ == '__main__':
    main()
