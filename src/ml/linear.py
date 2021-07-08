from sklearn.linear_model import SGDRegressor

from ml.common import GOAL_REWARD, INVALID_ACTION_REWARD
from .rl import RLAgent
from .common_functions import build_state, check_for_partial_goals, extract_objects, extract_offsets, build_state_linear
import numpy as np
from pddlgym.core import InvalidAction
import random
import time

class LinearQLearning(RLAgent):
    def __init__(self, env, obs, problem=0, action_list=None, check_partial_goals=True):
        self.predicates = env.observation_space.predicates
        self.objects = extract_objects(obs)
        self.check_partial_goals = check_partial_goals
        self.state_size, self.offsets = extract_offsets(self.objects['block'], self.predicates)
        #TODO: blocks will be hardcoded for now, change laters
        self.env = env
        self.discount = 0.99
        self.episodes = 10000
        self.timesteps = 50
        self.end_eps = 0.1
        self.start_eps = 0.9
        self.anneal_until = 1000000
        if action_list is not None:
            self.action_list = action_list
        else:
            self.action_list = list(env.action_space.all_ground_literals(obs, valid_only=False))
        self.num_actions = len(self.action_list)
        dummy_state = build_state_linear(obs, self.state_size, self.offsets, self.objects['block'], self.num_actions, random.randint(0, self.num_actions-1))
        # self.regressors = [SGDRegressor(learning_rate='constant').partial_fit(dummy_state, [np.random.normal()]) for _ in range(self.num_actions)]
        self.regressor = SGDRegressor().partial_fit(dummy_state, [0.])
        self.goal_literals_achieved = set()
        self.env.fix_problem_index(problem)
        self.problem = problem

    def eps(self, steps):
        return self.end_eps if steps >= self.anneal_until else \
            self.start_eps - ((self.start_eps - self.end_eps) /
            self.anneal_until * steps)

    def predict(self, state, a=None):
        if not a:
            states = []
            for i in range(self.num_actions):
                states.append(state[0].copy())
                states[i][self.state_size + i] = 1.
            # return [model.predict(state)[0] for model in self.regressors]
            return self.regressor.predict(states)
        else:
            state[0][self.state_size + a] = 1.
            return self.regressor.predict(state)
            # return self.regressors[a].predict(state)

    def next_action(self, state, steps=None):
        """
        returns the next best action
        defined by argmax(self.net.forward())
        state: an ndarray of the current state, preprocessed
        returns: the index of the best action
        """
        if steps:
            eps = self.eps(steps)
            # print(eps)
            if steps % 10000 == 0:
                print('current eps:', eps)
            # eps = self.eps_exp(steps)
            if random.random() <= eps:
                return random.randint(0, self.num_actions-1)
        q_values = self.predict(state)
        # print(q_values)
        # print(q_values)
        best_action = np.argmax(q_values)
        # print(best_action)
        return best_action

    def update(self, state, action, target):
        # self.regressors[action].partial_fit(state, [target])
        state[0][self.state_size + action] = 1.
        self.regressor.partial_fit(state, [target])
    
    def literal_from_vector(self, action):
        return self.action_list[action]

    def learn(self):
        print("initial coefficients:", self.regressor.coef_)
        training_step = 0
        max_r = float("-inf")
        for e in range(self.episodes):
            timestep = 0
            obs, _ = self.env.reset()
            done = False
            start_time = time.time()
            episode_r = 0.1
            episode_loss = 0.
            while timestep < self.timesteps and not done:
                
                state = build_state_linear(obs, self.state_size, self.offsets, self.objects['block'], self.num_actions)
                action = self.next_action(state, training_step)
                # print(action)
                literal_action = self.action_list[action]
                try:
                    obs, r, done, _ = self.env.step(literal_action)
                    if done:
                        print("GOAL FOUND!!!")
                        r = GOAL_REWARD
                    # else:
                    #     if self.check_partial_goals:
                    #         r += check_for_partial_goals(obs, self.goal_literals_achieved)
                except InvalidAction:
                    r = INVALID_ACTION_REWARD

                next_state = build_state_linear(obs, self.state_size, self.offsets, self.objects['block'], self.num_actions)

                q_values_next = self.predict(next_state)

                target = r + self.discount * np.max(q_values_next)
                # print(target)

                err = self.update(state, action, target)
                # episode_loss += err
                episode_r += r
                training_step += 1
                timestep += 1
            elapsed_time = time.time() - start_time
            timestep = 0
            if episode_r > max_r:
                max_r = episode_r
                print("New all-time high reward:", max_r)
            if (e + 1) % 1000 == 0:
                print(f'Episode {e+1} ended. Time to process: {elapsed_time}. Reward earned: {episode_r}. Episode loss: {episode_loss}. Avg. Q after episode: {0}. Current eps: {self.eps(training_step)}')
                print(f'Coefficients: {self.regressor.coef_}')
            self.goal_literals_achieved.clear()