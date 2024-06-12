import argparse
import os
import random
import time
from enum import Enum

import numpy as np
import torch
from eGCN_ppo import Agent as eGCN_agent
from moss import Engine, LaneChange, TlPolicy, Verbosity
from moss.agent import Agents
from moss.map import LaneType, Map
from tqdm import tqdm


class VehiclePolicy(Enum):
    SHORTEST = 1  # 不考虑道路价格
    RANDOM = 2  # 随机选取
    OPTIMUM = 3  # 选取综合代价最小的


_LOG = []
NN_INPUT_SCALER = 5


class Env:
    def __init__(
        self,
        map_file: str,
        agent_file: str,
        start_step: int,
        step_interval: int,
        step_reset: int,
        n_routes=3,
        fuel_cost_weight=0.0024*0.2,
        vehicle_policy=VehiclePolicy.SHORTEST,
    ):
        """
        step_interval:      CP算法行动间隔
        step_reset:         行动多少次后重置环境
        n_routes:           每个车辆的候选路径数目
        fuel_cost_weight:   车辆选择路径时会计算 代价=fuel_cost_weight*路径长度+路径价格
        """
        self.step_interval = step_interval
        self.step_reset = step_reset
        self.vehicle_policy = vehicle_policy
        self.start_step = self.time = start_step
        self.step_count = 0
        agents = Agents(agent_file).agents
        assert len(agents) % n_routes == 0
        self.map = Map(map_file)
        self.agents = [
            [
                agents[i].departure_time,
                [[i+j, [self.map.road_map[i].index for i in agents[i+j].route]] for j in range(n_routes)]
            ] for i in range(0, len(agents), n_routes)
        ]
        self.agents.sort(key=lambda x: -x[0])
        self._agents = self.agents[:]
        self.eng = Engine(
            map_file=map_file,
            agent_file=agent_file,
            start_step=start_step,
            verbose_level=Verbosity.NO_OUTPUT,
            lane_change=LaneChange.MOBIL,
            lane_veh_add_buffer_size=3000,
            lane_veh_remove_buffer_size=3000,
        )
        self.eng.set_tl_policy_batch(range(self.eng.junction_count), TlPolicy.FIXED_TIME)
        self.eng.set_tl_duration_batch(range(self.eng.junction_count), 30)
        self.eng.set_vehicle_enable_batch(range(self.eng.vehicle_count), False)
        self.road_prices = np.zeros(len(self.map.roads))
        l = np.array([r.lanes[0].geom.length for r in self.map.roads])
        self.road_fuel_cost = l*fuel_cost_weight
        self.r_2 = np.maximum(1, np.array([len(r.lanes) for r in self.map.roads]))
        self.r_3 = np.maximum(1, l)
        self.r_4 = np.maximum(1, l*self.r_2)
        self.road_travel_time = [[] for _ in range(self.eng.road_count)]
        self.road_free_time = np.array([r.lanes[0].length/r.lanes[0].max_speed for r in self.map.roads])
        self.lane2road = [l.parent_road.index if l.parent_road is not None else -1 for l in self.map.lanes]
        self.vehicle_enter_time = np.zeros(self.eng.vehicle_count)
        self.vehicle_lane = self.eng.get_vehicle_lanes()
        self._reset_id = self.eng.make_checkpoint()
        self.metrics = None
        self.obs_size = (self.eng.road_count, 4)
        self.info = {
            'ATT-d': 1e999,
            'ATT-f': 1e999,
            'Throughput': 0,
            'reward': 0,
        }

    def _reset(self):
        self.time = self.start_step
        self.step_count = 0
        self.agents = self._agents[:]
        self.road_prices[:] = 0
        for i in self.road_travel_time:
            i.clear()
        self.eng.restore_checkpoint(self._reset_id)

    def _step(self):
        # 处理agent，到时间则放行
        while self.agents and self.agents[-1][0] <= self.time+1:
            irs = self.agents.pop()[1]
            if self.vehicle_policy == VehiclePolicy.SHORTEST:
                choice = irs[0][0]
            elif self.vehicle_policy == VehiclePolicy.RANDOM:
                choice = random.choice(irs)[0]
            else:
                ic = [[i, self.road_fuel_cost[r].sum(), self.road_prices[r].sum()] for i, r in irs]
                for (_, a, b), (_, r) in zip(ic, irs):
                    _LOG.append([self.time, a, b, len(r)])
                choice = min(ic, key=lambda x: x[1]+x[2])[0]
            self.eng.set_vehicle_enable(choice, True)
        # 处理road，记录平均通行时间
        vl = self.eng.get_vehicle_lanes()
        mask = vl != self.vehicle_lane
        if np.any(mask):
            for i, lane, t in zip(np.nonzero(mask)[0], self.vehicle_lane[mask], self.vehicle_enter_time[mask]):
                if lane != -1 and self.lane2road[lane] != -1:
                    self.road_travel_time[self.lane2road[lane]].append(self.time-t)
                self.vehicle_enter_time[i] = self.time
        self.vehicle_lane = vl

    def step(self):
        for _ in range(self.step_interval):
            self._step()
            self.eng.next_step()
            self.time = self.eng.get_current_time()
        self.step_count += 1
        if self.step_count == self.step_reset:
            obs = self.get_obs()
            self.metrics = self.get_metrics()
            self._reset()
            return True, obs
        return False, self.get_obs()

    def step_ppo(self, action):
        self.road_prices[:] = action  # *self.road_fuel_cost
        # r = self.eng.get_finished_vehicle_count()
        # r = self.eng.get_departed_vehicle_average_traveling_time()
        r = self.eng.get_vehicle_total_distances().sum()
        if not np.isfinite(r):
            r = 0
        for _ in range(self.step_interval):
            self._step()
            self.eng.next_step()
            self.time = self.eng.get_current_time()
        # v = self.eng.get_vehicle_speeds()
        # v = v[v >= 0]
        # r = (np.mean(v)-10)/10 if len(v) else 0
        # r = (self.eng.get_finished_vehicle_count()-r)/50
        # r = (r-self.eng.get_departed_vehicle_average_traveling_time())/10
        r = (self.eng.get_vehicle_total_distances().sum()-r)/1e6
        if not np.isfinite(r):
            r = 0
        self.info['reward'] = r
        s = self.observe()
        self.step_count += 1
        done = False
        if self.step_count == self.step_reset:
            self.info['ATT-d'] = self.eng.get_departed_vehicle_average_traveling_time()
            self.info['ATT-f'] = self.eng.get_finished_vehicle_average_traveling_time()
            self.info['Throughput'] = self.eng.get_finished_vehicle_count()
            self._reset()
            done = True
            s = self.observe()
        return s, r, done, self.info

    def get_metrics(self):
        return (
            self.eng.get_running_vehicle_count(),
            self.eng.get_finished_vehicle_count(),
            self.eng.get_finished_vehicle_average_traveling_time(),
            self.eng.get_departed_vehicle_average_traveling_time(),
        )

    def generate_graph(self):
        # 以每个road为结点，连接关系为边
        node_count = self.eng.road_count
        edges = []
        for j in self.map.junctions:
            for l in j.lanes:
                if l.type == LaneType.DRIVING and l.predecessors and l.successors:
                    edges.append((
                        l.predecessors[0].parent_road.index,
                        l.successors[0].parent_road.index,
                    ))
        edges = sorted(set(edges))
        return (
            node_count,
            torch.tensor(edges, dtype=torch.long).T.contiguous()
        )

    def get_accumulated_reward(self):
        # 累计奖励为完成的车辆数
        return self.eng.get_finished_vehicle_count()/100

    def get_obs(self):
        # 观测值为道路的车辆数
        return self.eng.get_road_vehicle_counts()/100

    def observe(self):
        # 用于我们PPO训练的观测值
        c = self.eng.get_road_vehicle_counts()
        return np.stack([
            np.minimum(c/200, 1)*NN_INPUT_SCALER,
            np.minimum(c/self.r_2/50, 1)*NN_INPUT_SCALER,
            np.minimum(c/self.r_3*3, 1)*NN_INPUT_SCALER,
            np.minimum(c/self.r_4*3, 1)*NN_INPUT_SCALER,
        ]).T


class NoController:
    def __init__(self, env):
        self.env = env

    def step(self):
        self.env.step()


class DeltaController:
    def __init__(self, env: Env, R: float, beta: float, tau: float = 5, N=10):
        self.env = env
        self.R = R
        self.beta = beta
        self.N = N
        self.env.road_prices[:] = tau

    def step(self):
        dt = np.array([
            np.mean(t[-self.N:])-t0 if t else 0
            for t, t0 in zip(self.env.road_travel_time, self.env.road_free_time)
        ])
        dt = np.maximum(0, dt)
        self.env.road_prices[:] = (1-self.R)*self.env.road_prices+self.R*self.beta*dt
        self.env.step()


class EGCNController:
    def __init__(self, env: Env, args, max_price=1):
        self.env = env
        self.args = args
        self.max_price = max_price
        self.model = eGCN_agent(self.env, args)
        self.obs = self.env.get_obs()
        self.actor_loss = self.critic_loss = 0

    def step(self):
        action = self.model.get_action(self.obs)
        self.env.road_prices[:] = action*self.max_price
        done, next_obs = self.env.step()
        if done:
            self.obs = self.env.get_obs()
        else:
            self.obs = next_obs
        if self.args.egcn_train:
            reward = self.model.get_rewards()
            self.model.add_sample_and_train(reward, next_obs, done)
            if self.model.loss:
                self.critic_loss, self.actor_loss = self.model.loss[-1]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="name of the experiment")
    parser.add_argument('--data', type=str, default='../data/hangzhou_cp')
    parser.add_argument('--algo', choices='none random deltatoll eGCN'.split(), required=True)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--interval', type=int, default=20)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--steps', type=int, default=7200)
    parser.add_argument('--suffix', type=str)
    # deltatoll
    parser.add_argument('--dt_R', type=float, default=0.04144857024110564)
    parser.add_argument('--dt_beta', type=float, default=0.03470888810354356)
    # eGCN
    parser.add_argument('--egcn_lr', type=float, default=3e-4)
    parser.add_argument('--egcn_train_epochs', type=int, default=10000)
    parser.add_argument('--egcn_mini_batch_size', type=int, default=32)  # 64
    parser.add_argument('--egcn_mini_batch_num', type=int, default=8)  # 16
    parser.add_argument('--egcn_update_epochs', type=int, default=10)
    parser.add_argument('--egcn_checkpoint', type=str)

    args = parser.parse_args()
    args.reset = args.steps//args.interval
    args.egcn_train = args.algo == 'eGCN'

    set_seed(args.seed)
    torch.set_float32_matmul_precision('medium')

    if args.exp is None:
        path = time.strftime(f'log/{args.algo}/%Y%m%d-%H%M%S')
    else:
        path = time.strftime(f'log/{args.algo}/{args.exp}/%Y%m%d-%H%M%S')
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if args.algo == 'none':
        policy = VehiclePolicy.SHORTEST
    elif args.algo == 'random':
        policy = VehiclePolicy.RANDOM
    else:
        policy = VehiclePolicy.OPTIMUM
    env = Env(
        map_file=f'{args.data}/map.bin',
        agent_file=f'{args.data}/agents.bin',
        start_step=args.start,
        step_interval=args.interval,
        step_reset=args.reset+int(not args.egcn_train),
        vehicle_policy=policy,
    )
    if args.algo == 'deltatoll':
        controller = DeltaController(env, R=args.dt_R, beta=args.dt_beta, tau=0.2)
    elif args.algo == 'eGCN':
        controller = EGCNController(env, args)
        t = time.time()
        for s in tqdm(range(args.reset*args.egcn_train_epochs), ncols=100):
            controller.step()
            if (s+1) % args.reset == 0:
                with open(f'{path}/info.log', 'a') as f:
                    f.write(f"{env.metrics[3]:.3f} {env.metrics[1]} {time.time()-t:.3f}\n")
                    t = time.time()
        return
    else:
        controller = NoController(env)

    t = time.time()
    for _ in tqdm(range(args.reset), ncols=100):
        controller.step()
    att = env.eng.get_departed_vehicle_average_traveling_time()
    tp = env.eng.get_finished_vehicle_count()
    print(f"ATT: {att:.3f} TP: {tp}")
    with open(f'{path}/info.log', 'a') as f:
        f.write(f"{att:.3f} {tp} {time.time()-t:.3f}\n")


if __name__ == '__main__':
    main()
