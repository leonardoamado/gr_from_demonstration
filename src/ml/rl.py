import pickle
import numpy as np
import random

from pddlgym.core import InvalidAction

# This will be an implementation of Q-Learning with Gym

# use epsilon-greedy policies instead of 1 for maximum, 0 otherwise
# test with negative rewards
# policy gradient

class TabularQLearner:
    def __init__(self,
                 env,
                 init_obs,
                 problem=None,
                 episodes=25000,
                 decaying_eps=True,
                 eps=0.9,
                 alpha=0.01,
                 decay=0.000005,
                 gamma=0.99,
                 action_list=None):
        self.env = env
        if not action_list:
            self.action_list = list(env.action_space.all_ground_literals(init_obs, valid_only=False))
        else:
            self.action_list = action_list
        self.actions = len(self.action_list)

        self.q_table = {}

        # hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.decay = decay
        self.c_eps = eps
        self.base_eps = eps
        if decaying_eps:

            def epsilon():
                self.c_eps = max(self.c_eps - self.decay, 0.01)

                return self.c_eps

            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha

        if problem:
            self.env.fix_problem_index(problem)


    def save_q_table(self, path):
        with open(path, 'w') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path):
        with open(path, 'r') as f:
           table = pickle.load(path)
           self.q_table = table

    def add_new_state(self, state):
        self.q_table[state] = [0 for _ in range(self.actions)]

    def best_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0 for _ in range(self.actions)]
        return np.argmax(self.q_table[state])


    def get_max_q(self, state):
        if state not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[state])

    def set_q_value(self, state, a, q):
        if state not in self.q_table:
            self.add_new_state(state)
        self.q_table[state][a] = q

    def get_q_value(self, state, a):
        if state not in self.q_table:
            self.add_new_state(state)
        return self.q_table[state][a]

    def learn(self):
        tsteps = 50
        for n in range(self.episodes):
            state, info = self.env.reset()
            done = False
            tstep = 0
            while tstep < tsteps and not done:
                # print(self.eps)
                eps = self.eps()
                # print(eps)
                if random.random() <= eps:
                    a = random.randint(0, self.actions-1)
                    # a = self.env.action_space.sample(state)
                else:
                    a = self.best_action(state)
                try:
                    next_state, r, done, _ = self.env.step(self.action_list[a])
                except InvalidAction:
                    next_state = state
                    r = 0.
                    done = False
                # if done:
                #     print('Found a terminal state after ', + tstep, ' time steps.')

                next_max_q = self.get_max_q(next_state)
                old_q = self.get_q_value(state, a)
                # old_q = self.q_table[state][a]

                new_q = old_q + self.alpha * \
                    (r + (self.gamma * next_max_q) - old_q)
                # if done:
                #     self.set_q_value(state, a, 1)
                # else:
                self.set_q_value(state, a, new_q)
                # self.q_table[state][a] = new_q
                state = next_state
                tstep += 1
            if n % 1000 == 0:
                print(f'Episode finished. Timestep: {tstep}. Done: {done}')
        print(len(self.q_table.keys()))