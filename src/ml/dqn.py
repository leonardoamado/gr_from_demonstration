import torch
import operator
import random
import numpy as np
from collections import deque
from pddlgym.core import InvalidAction
import time
import os
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
import datetime
from functools import reduce
from matplotlib import pyplot as plt
import math

from ml.base import BaseMethod
from ml.memories.inmemory_replay import InMemoryReplay
import ml.common


class MLP(torch.nn.Module):
    def __init__(self, state_size, num_actions, start_eps=0.9, end_eps=0.1, dueling=True, dueling_type='mean'):
        super(MLP, self).__init__()
        self.state_size = state_size
        self.dueling = dueling
        self.device = torch.device(f"cuda" if torch.cuda.is_available()
                                   else "cpu")
        self.fc1 = torch.nn.Linear(state_size, 128)
        self.fc2 = torch.nn.Linear(128, 256)
        self.fc3 = torch.nn.Linear(256, 256)
        if dueling:
            self.fc_value = torch.nn.Linear(256, 1)
            self.fc_advantages = torch.nn.Linear(256, num_actions)
            if dueling_type == 'max':
                self.dueling_agg = torch.mean
            elif dueling_type == 'mean':
                self.dueling_agg = torch.max
        self.out = torch.nn.Linear(128, num_actions)
        self.num_actions = num_actions
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.anneal_until = 120000
        self.gamma = 0.99

        self.to(self.device)
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        if self.dueling:
            adv = F.relu(self.fc_advantages(x))
            val = F.relu(self.fc_value(x))
            adv_agg = self.dueling_agg(adv)
            x = val + adv - adv_agg
            pass
        else:
            x = self.out(x)
        x = torch.squeeze(x)
        return x

    def target_q(self, qs, qs_p, qs_p_target, a, r, t):
        f_q = np.copy(qs)
        qs_p = np.squeeze(qs_p)
        qs = np.squeeze(qs)
        q = np.argmax(qs_p, axis=1)
        f_q = np.squeeze(f_q)
        qs_p_target = np.squeeze(qs_p_target)
        for _i, done in enumerate(t):
            _r = r[_i]
            _a = a[_i]
            # temporary fix for
            # IndexError: index 33 is out of bounds for axis 1 with size 1
            # squeeze tensor, removing 1-sized dimensions

            d = qs_p_target[_i, q[_i]]
            f_q[_i, _a] = _r if done else _r + qs_p_target[_i, q[_i]] * self.gamma
        return f_q

    def train(self, batch, target=None):
        # memory is built as State x Action x Next State x Reward x Is Terminal
        s, a, s_p, r, t = batch[0], batch[1], batch[2], batch[3], batch[4]
        with torch.no_grad():
            next_state_torch = torch.from_numpy(s_p).type(torch.FloatTensor).to(self.device)
            qs_future = self.forward(next_state_torch)
            qs_future_numpy = qs_future.cpu().data.numpy()
        self.optimizer.zero_grad()
        state_torch = torch.from_numpy(s).type(torch.FloatTensor).to(self.device)
        qs = self.forward(state_torch)
        qs_numpy = qs.cpu().data.numpy()
        f_q = torch.from_numpy(self.target_q(qs_numpy, qs_future_numpy, qs_future_numpy, a, r, t)).to(self.device)
        # print(f_q)
        loss = self.loss(f_q.float(), qs.float())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def next_action(self, state, steps=None, q=False):
        """
        returns the next best action
        defined by argmax(self.net.forward())
        state: an ndarray of the current state, preprocessed
        returns: the index of the best action
        """
        if steps:
            eps = self.eps(steps)
            # eps = self.eps_exp(steps)
            if random.random() <= eps:
                return random.randint(0, self.num_actions-1)
        qs = self.forward(state)
        best_action = torch.argmax(qs, axis=0).cpu().detach().numpy()
        if q:
            return qs, best_action
        else:
            return best_action

    def eps(self, steps):
        return self.end_eps if steps >= self.anneal_until else \
            self.start_eps - ((self.start_eps - self.end_eps) / self.anneal_until * steps)

    def parameter_count(self):
        return sum(map(lambda x: reduce(operator.mul, x.shape, 1), self.parameters()))


class DQN(BaseMethod):

    def __init__(self, env, state, problem=0, params=None, action_list=None, check_partial_goals=True):
        # self, params, actions, input_shape=(4, 64, 64)):
        self.params = params if params is not None \
            else ml.common.DEFAULT_PARAMS
        self.env = env
        self.env.fix_problem_index(problem)
        self.problem = problem
        self.batch_size = self.params['batch_size']
        self.average_qs = []
        self.max_timesteps = 50
        self.number_of_skips = 5
        self.check_partial_goals = check_partial_goals
        self.goal_literals_achieved = set()

        self.predicates = env.observation_space.predicates

        self.blocks = ['d', 'r', 'a', 'w', 'o', 'e', 'p', 'c']

        if action_list is None:
            self.action_list = list(env.action_space.all_ground_literals(state, valid_only=False))
        else:
            self.action_list = action_list

        self.extract_offsets()
        self.net = MLP(self.state_size, len(action_list))
        self.memory = InMemoryReplay(size=self.params['mem_size'], input_shape=self.state_size)
        self.test_memory = InMemoryReplay(size=self.params['dry_size'], input_shape=self.state_size)
        self.curr_state = deque(maxlen=self.params['history'])
        self.next_state = deque(maxlen=self.params['history'])
        # self.net = self.net.to(self.net.device)
        # self.train_skip = 8
        # print(f'RND has {self.rnd.count_parameters()} parameters.')

    def clear_memories(self):
        self.memory.clear()
        self.test_memory.clear()

    def build_state(self, obs):
        state = [0. for _ in range(self.state_size)]
        ground_literals = obs[0]
        for lit in ground_literals:
            base_offset = self.offsets[lit.predicate.name]
            var_offset = 1
            vars = lit.variables
            if len(vars) == 2:
                idx_first = self.blocks.index(vars[0].name)
                idx_second = self.blocks.index(vars[1].name)
                var_offset = len(self.blocks) * idx_first + idx_second
            else:
                v = vars[0]
                if v.var_type == 'block':
                    var_offset = self.blocks.index(v.name)
                else:
                    var_offset = 0
            # for v in lit.variables:
            #     if v.var_type == 'block':
            #         if len(lit.variables) == 2:
            #             var_offset = ?
            #         # var_offset *= self.blocks.index(v.name)
            #     else:
            #         var_offset = 0
            # print('Offsets:',base_offset, var_offset, base_offset + var_offset)
            state[base_offset + var_offset] = 1.
        return torch.FloatTensor(state)

    def extract_offsets(self):
        offsets = {}
        i = 0
        for pred in self.predicates:
            offsets[pred.name] = i
            num_lits = 1
            for t in pred.var_types:
                if t == 'block':
                    num_lits *= len(self.blocks)
            i += num_lits
        self.state_size = i
        self.offsets = offsets

    def literal_from_vector(self, action):
        return self.action_list[action]

    def loss(self, qs, qs_p, qs_p_target, a, r, t):
        # torch.from_numpy
        return self.future_q(qs, qs_p, qs_p_target, a, r, t)
        # return [qs[i] if i != a else r for i in range(self.n_actions)]

    def future_q(self, qs, qs_p, qs_p_target, a, r, t):
        f_q = np.copy(qs)
        q = np.argmax(qs_p, axis=1)
        for _i, _t in enumerate(t):
            _r = r[_i]
            _a = a[_i]
            f_q[_i, _a] = _r if _t else _r + qs_p_target[_i, q[_i]] * self.gamma
        return f_q

    def dry_run(self):

        for episode in range(self.params['episodes']):
            pass

    def learn(self):
        training_steps = 1
        skip = 0
        print(f'Training model {type(self.net).__name__}. Parameters: {self.net.parameter_count():,d}.')
        # self.dry_run(self.params['dry_size'])
        writer = self.create_tensorboard()
        max_reward = float("-inf")
        # s, _, _, _, _ = self.test_memory.get_batch(10)
        # s = torch.from_numpy(s).to(self.net.device)
        # writer.add_graph(self.net, input_to_model=s, verbose=True)
        for episode in range(self.params['episodes']):
            episode_loss = 0.
            episode_r = 0.

            obs, _ = self.env.reset()
            done = False
            start_time = time.time()
            timestep = 0
            while timestep < self.max_timesteps and not done:

                state = self.build_state(obs)
                # self.curr_state.
                _a = self.net.next_action(state, training_steps)
                if _a >= 80:
                    print('adding this fucking thing', _a)
                action = self.literal_from_vector(_a)
                # print(action)
                # r = self.apply_action(a)
                # i_r = self.rnd_reward(s).detach().clamp(-1., 1.).item()
                # r = self.normalize_reward(r)
                # r_combined = r + i_r
                # print(r_combined)

                try:

                    obs, r, done, _ = self.env.step(action)
                    if done:
                        print('GOAL REACHED POGGERS')
                        r = ml.common.GOAL_REWARD
                    else:
                        r = ml.common.TIMESTEP_REWARD
                        if self.check_partial_goals:
                            r += self.check_for_partial_goals(obs)
                except InvalidAction:
                    r = ml.common.INVALID_ACTION_REWARD
                next_state = self.build_state(obs)
                # if done:
                #     next_state = state
                # else:
                #     next_state = self.doom.get_state().screen_buffer

                # s_p = self.state_to_net_state(next_state, self.next_state)

                self.memory.add_transition(state, _a, next_state, r, done)

                if skip == self.number_of_skips:
                    batch = self.memory.get_batch(self.batch_size)
                    if not batch:
                        continue

                    # loss = self.net.train(batch, self.target_net)
                    loss = self.net.train(batch)
                    episode_loss += loss
                    # self.train_rnd(batch[0])
                    training_steps += 1
                    skip = 0
                else:
                    skip += 1

                timestep += 1

                # if training_steps % 5000 == 0:
                #     self.target_net.load_state_dict(self.net.state_dict())

                # if training_steps % 10000 == 0:
                #     self.serialize_model(training_steps)
                episode_r += r

            elapsed_time = time.time() - start_time
            if episode_r > max_reward:
                max_reward = episode_r
                print("new max reward reached:", max_reward)
            # avg_q = self.average_q_test()
            # self.write_tensorboard(writer, episode_loss, episode_r, avg_q)
            # self.average_qs.append(avg_q)
            if (episode + 1) % 1000 == 0:
                print(f'Episode {episode} ended. Time to process: {elapsed_time}. Reward earned: {episode_r}. Episode loss: {episode_loss}. Avg. Q after episode: {0}. Current eps: {self.net.eps(training_steps)}')

            self.curr_state.clear()
            self.next_state.clear()
            self.goal_literals_achieved.clear()
        self.clear_memories()

    def check_for_partial_goals(self, obs):
        literals = obs[0]
        goals = obs[2].literals
        r = 0.
        for lit in literals:
            if lit not in self.goal_literals_achieved and lit in goals:
                r += ml.common.PARTIAL_GOAL_REWARD
                self.goal_literals_achieved.add(lit)
        return r

    def create_tensorboard(self):
        src_dir = os.environ['RLROOT']
        scenario = self.env.domain.domain_name
        log_path = f'{src_dir}/logs/{scenario}/{type(self.net).__name__}/try_{time.time()}'
        os.makedirs(log_path, exist_ok=True)
        return SummaryWriter(log_dir=log_path)

    def write_tensorboard(self, w, l, r, q):
        w.add_scalar('Reward per episode', r)
        w.add_scalar('Avg Q per episode', q)
        w.add_scalar('Loss per episode', l)
        w.flush()

    def normalize_reward(self, r):
        return r

    def average_q_test(self):
        qs = np.zeros((self.test_memory.max_size))
        for i in range(0, len(self.test_memory.s), 32):
            end = min(i + 32, self.test_memory.curr)
            states = torch.FloatTensor(self.test_memory.s[i:end])
            # s_net = self.net.to_net(s)
            result = self.net.forward(states)
            qs[i:end] = torch.max(self.net.forward(states), axis=1)[0].cpu().data.numpy()
        qs = np.sum(qs) / self.test_memory.max_size
        return qs

    def serialize_model(self, steps):
        base_dir = os.environ['VZD_TORCH_DIR']
        scenario = self.params['doom_config'].split('/')[-1].split('.')[0]
        scenario_dir = f'{base_dir}/weights/{scenario}'
        os.makedirs(scenario_dir, exist_ok=True)
        path = f'{scenario_dir}/{self.net.name}_{scenario}_{time.time()}.pt'
        torch.save(self.net.state_dict(), path)

    def process_state(self,
                      obs,
                      epsilon=0.,
                      distribution='softmax',
                      action=True):
        MIN_VAL = 1e-6

        state = self.build_state(obs)
        q, a = self.net.next_action(state, q=True)
        if distribution:
            q = DQN.build_distribution(q, distribution, epsilon)
        if action:
            return q, a
        else:
            return q

    @staticmethod
    def build_distribution(q, distribution_type, epsilon):
        if distribution_type == 'softmax':
            return DQN.softmax(q)
        elif distribution_type == 'epsilon_greedy':
            num_actions = len(q)
            e_greedy = [1e-6 + epsilon/num_actions for _ in range(len(q))]
            e_greedy[np.argmax(q)] = 1 - epsilon - 1e-6 * num_actions
            return e_greedy

    @staticmethod
    def softmax(qs):
        return [(math.exp(q))/sum([math.exp(_q) for _q in qs]) for q in qs]

    # def build_action(self, a):
    #     return [1 if a == i else 0 for i in range(self.net.actions)]

    # def rnd_reward(self, s):
    #     s_rnd = self.net.to_net(s)
    #     f_rnd = self.rnd.forward(s_rnd)
    #     with torch.no_grad():
    #         f_target = self.target_rnd.forward(s_rnd).detach()
    #     return torch.pow(f_target - f_rnd, 2).sum()

    # def train_rnd(self, s):
    #     self.rnd.optim.zero_grad()
    #     t = self.rnd_reward(s)
    #     # y_pred = self.rnd.forward(s_rnd)
    #     # y_true = self.target_rnd.forward(s_rnd)
    #     t.backward()
    #     self.rnd.optim.step()
