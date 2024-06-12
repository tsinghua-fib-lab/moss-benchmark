import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.beta import Beta
from torch.optim import Adam
from torch_geometric.nn import GCNConv

LOG_STD_MAX = 2
LOG_STD_MIN = -20
epsilon = 1e-6


class eGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(-1, 16)
        self.conv2 = GCNConv(16, 16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.tanh(x)
        x = self.conv2(x, edge_index)
        x = F.tanh(x)
        return x


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


class Model(nn.Module):
    def __init__(self, n_node, node_dim, edge_index):
        super().__init__()
        self.register_buffer('edge_index', edge_index, False)

        self.conv1 = GCNConv(-1, node_dim)
        self.conv2 = GCNConv(node_dim, node_dim)

        self.critic = make_mlp(n_node*node_dim, 64, 64, 1, sigma=1)
        self.actor_alpha = make_mlp(n_node*node_dim, 128, 128, n_node, sigma=0.01)
        self.actor_beta = make_mlp(n_node*node_dim, 128, 128, n_node, sigma=0.01)

    def get_obs(self, x):
        # x: B x N x F
        x = self.conv1(x, self.edge_index)
        x = F.tanh(x)
        x = self.conv2(x, self.edge_index)
        x = F.tanh(x)
        # B x (N*D)
        return x.view(x.size(0), -1)

    def get_value(self, x):
        # B x (N*D)
        return self.critic(self.get_obs(x))

    def get_action_and_value(self, x, action=None, sample=True):
        x = self.get_obs(x)
        alpha = 1+F.softplus(self.actor_alpha(x))
        beta = 1+F.softplus(self.actor_beta(x))
        probs = Beta(alpha, beta)
        if action is None:
            if sample:
                action = probs.sample()
            else:
                action = alpha / (alpha + beta)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


class Agent:
    def __init__(self, world, args):
        self.node_num, self.edge_index = world.generate_graph()
        self.world = world
        self.args = args
        self.loss = []
        args.num_steps = args.egcn_mini_batch_size*args.egcn_mini_batch_num
        self.last_reward = world.get_accumulated_reward()

        self.device = device = torch.device('cuda')
        self.model = Model(self.node_num, 16, self.edge_index).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=args.egcn_lr, eps=1e-5)
        self.step = 0
        self.obs = torch.zeros((args.num_steps, self.node_num, 1), device=device)
        self.actions = torch.zeros((args.num_steps, self.node_num), dtype=torch.long, device=device)
        self.log_probs = torch.zeros((args.num_steps, self.node_num), device=device)
        self.rewards = torch.zeros((args.num_steps, 1), device=device)
        self.dones = torch.zeros((args.num_steps, 1), device=device)
        self.values = torch.zeros((args.num_steps, 1), device=device)
        self.next_obs = None
        self.next_done = None

    @torch.no_grad
    def get_action(self, obs):
        if self.next_obs is None:
            self.next_obs = torch.FloatTensor(obs).view(-1, 1).to(self.device)
            self.next_done = torch.FloatTensor([0]).to(self.device)
        self.obs[self.step] = self.next_obs
        self.dones[self.step] = self.next_done
        action, log_prob, _, value = self.model.get_action_and_value(self.next_obs.unsqueeze(0))
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.values[self.step] = value
        return action.view(-1).cpu().numpy()

    def get_rewards(self):
        current_reward = self.world.get_accumulated_reward()
        reward = current_reward-self.last_reward
        self.last_reward = current_reward
        return np.array([reward])

    def add_sample_and_train(self, reward, next_obs, next_done):
        self.rewards[self.step] = reward.item()
        self.next_obs = torch.FloatTensor(next_obs).view(-1, 1).to(self.device)
        self.next_done = torch.FloatTensor([next_done]).to(self.device)
        self.step += 1
        if self.step == self.args.num_steps:
            self.update_parameters()
            self.step = 0

    def update_parameters(self):
        GAMMA = 0.99
        LAMBDA = 0.9
        CLIP_COEF = 0.2
        with torch.no_grad():
            next_value = self.model.get_value(self.next_obs.unsqueeze(0))
            advantages = torch.zeros_like(self.rewards)
            last_gae_lam = 0
            for t in reversed(range(self.args.num_steps)):
                if t == self.args.num_steps - 1:
                    next_nonterminal = 1.0 - self.next_done
                    next_values = next_value
                else:
                    next_nonterminal = 1.0 - self.dones[t + 1]
                    next_values = self.values[t + 1]
                delta = self.rewards[t] + GAMMA * next_values * next_nonterminal - self.values[t]
                advantages[t] = last_gae_lam = delta + GAMMA*LAMBDA * next_nonterminal * last_gae_lam
            returns = advantages + self.values

        b_obs = self.obs
        b_log_probs = self.log_probs
        b_actions = self.actions
        b_advantages = advantages
        b_returns = returns
        b_values = self.values
        clipfracs = []
        for epoch in range(self.args.egcn_update_epochs):
            b_inds = np.arange(self.args.num_steps)
            np.random.shuffle(b_inds)
            b_inds = b_inds.reshape(-1, self.args.egcn_mini_batch_size)
            for mb_inds in b_inds:
                _, new_log_prob, entropy, new_value = self.model.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                if np.any(log_ratio.detach().cpu().numpy() >= 80):
                    print('Warning: log_ratio too big', log_ratio)
                    log_ratio = torch.clamp(log_ratio, None, 80)
                ratio = log_ratio.exp()

                # with torch.no_grad():
                #     # calculate approx_kl http://joschu.net/blog/kl-approx.html
                #     old_approx_kl = (-log_ratio).mean()
                #     approx_kl = ((ratio - 1) - log_ratio).mean()
                #     clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                # if args.norm_adv:
                #     mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # entropy_loss = entropy.mean()
                # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
                loss = pg_loss + v_loss * 0.5

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                # self.loss.append([v_loss.item(), pg_loss.item()])
                # np.save('_egcn_loss.npy', np.array(self.loss))
