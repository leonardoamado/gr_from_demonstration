from abc import abstractmethod
import pickle
from typing import Any, Collection
from gym.core import Env
import numpy as np
import random
import datetime

from random import Random

from pddlgym.core import InvalidAction
from pddlgym.structs import Literal

from ml.common import GOAL_REWARD
from ml.common_functions import check_for_partial_goals
from utils import solve_fset
from pddlgym_planners.fd import FD


# This will be an implementation of Q-Learning with Gym

# use epsilon-greedy policies instead of 1 for maximum, 0 otherwise
# test with negative rewards
# policy gradient

# test value iteration with one traj after learning policies (later)

# hyperparameters tested

# run 1
#   episodes: 60k
#   decay: 0.0000002
#   timesteps: 100
#   alpha: 0.01
#   gamma: 0.99
#   patience: 40 (needed?)
#   1, -1, -10
#   found goal once in some thousand episodes
#   the -1 for every timestep makes the agent search several states
#
# run 2
#   episodes: 60k
#   decay: 0.0000002
#   timesteps: 100
#   alpha: 0.01
#   gamma: 0.99
#   patience: 40 (needed?)
#   1, 0, -10
#
#   searches a small space of states because of 0 on every timestep
#
# run 3
#   episodes: 60k
#   decay: 0.0000003
#   timesteps: 50
#   alpha: 0.01
#   gamma: 0.85
#   patience: inf
#   1, -1, -10


class RLAgent:
    """
    This is a base class used as parent class for any
    RL agent. This is currently not much in use, but is
    recommended as development goes on.
    """
    def __init__(self,
                 env: Env,
                 problem: int = None,
                 episodes: int = 100,
                 decaying_eps: bool = True,
                 eps: float = 0.9,
                 alpha: float = 0.01,
                 decay: float = 0.00005,
                 gamma: float = 0.99,
                 action_list: Collection[Literal] = None,
                 rand: Random = Random()):
        self.env = env
        self.problem = problem
        self.episodes = episodes
        self.decaying_eps = decaying_eps
        self.eps = eps
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.action_list = action_list
        self._random = rand
        if problem:
            self.env.fix_problem_index(problem)

    @abstractmethod
    def agent_step(self, reward: float, state: Any) -> Any:
        """A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state observation from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        pass

    @abstractmethod
    def policy(self, state: Any) -> Any:
        """The action for the specified state under the currently learned policy
           (unlike agent_step, this does not update the policy using state as a sample.
           Args:
                state (Any): the state observation from the environment
           Returns:
                The action prescribed for that state
        """
        pass


class TabularQLearner(RLAgent):
    """
    A simple Tabular Q-Learning agent.
    """
    def __init__(self,
                 env: Env,
                 init_obs: Any,
                 problem: int = None,
                 episodes: int = 10000,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 action_list: Collection[Literal] = None,
                 rand: Random = Random(),
                 check_partial_goals: bool = True,
                 valid_only: bool = False,
                 **kwargs):
        super().__init__(env, problem=problem, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, action_list=action_list, rand=rand)
        self.valid_only = valid_only
        if not action_list:
            self.action_list = list(env.action_space.all_ground_literals(init_obs, valid_only=False))
        else:
            self.action_list = action_list
        self.actions = len(self.action_list)
        self.check_partial_goals = check_partial_goals
        self.goal_literals_achieved = set()

        self.q_table = {}

        # hyperparameters
        self.episodes = episodes
        self.gamma = gamma
        self.decay = decay
        self.c_eps = eps
        self.base_eps = eps
        self.patience = 400000
        if decaying_eps:

            def epsilon():
                self.c_eps = max(self.c_eps - self.decay, 0.1)

                return self.c_eps

            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha

    def policy(self, state: Any) -> Any:
        return self.best_action(solve_fset(state))

    def agent_step(self, reward: float, state: Any) -> Any:
        # TODO We should definitely implement this better
        return self.best_action(solve_fset(state))

    def save_q_table(self, path: str):
        # sadly, this does not work, because the state we are using
        # is a frozenset of literals, which are not serializable.
        # a way to fix this is to use array states built using
        # common_functions.build_state

        with open(path, 'w') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, path: str):
        with open(path, 'r') as f:
            table = pickle.load(path)
            self.q_table = table

    def add_new_state(self, state: Any):
        self.q_table[state] = [1. for _ in range(self.actions)]

    def best_action(self, state: Any):
        if state not in self.q_table:
            self.add_new_state(state)
            # self.q_table[state] = [0 for _ in range(self.actions)]
        return np.argmax(self.q_table[state])

    def get_max_q(self, state: Any):
        if state not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[state])

    def set_q_value(self, state: Any, action: Any, q_value: float):
        if state not in self.q_table:
            self.add_new_state(state)
        self.q_table[state][action] = q_value

    def get_q_value(self, state: Any, action: Any) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return self.q_table[state][action]

    def learn(self, forced_init: bool = True, init_threshold: int = 20):
        log_file = f'logs/tabular_q_learning_{datetime.datetime.now()}'
        tsteps = 50
        done_times = 0
        patience = 0
        converged_at = None
        max_r = float("-inf")
        init, _ = self.env.reset()
        planner = FD()
        init, _ = self.env.reset()
        plan = planner(self.env.domain, init)
        print(len(plan))
        print('LEARNING FOR GOAL:', init.goal)
        for n in range(self.episodes):
            episode_r = 0
            state, info = self.env.reset()
            state = solve_fset(state.literals)
            done = False
            tstep = 0
            while tstep < tsteps and not done:
                eps = self.eps()
                if forced_init and n < init_threshold and tstep < (len(plan)):
                    action = self.action_list.index(plan[tstep])
                    # print('Forced step:', action, tstep)
                else:
                    if self._random.random() <= eps:
                        action = self._random.randint(0, self.actions-1)
                        # a = self.env.action_space.sample(state)
                    else:
                        action = self.best_action(state)
                try:
                    obs, reward, done, _ = self.env.step(self.action_list[action])
                    next_state = solve_fset(obs.literals)
                    if done:
                        reward = 100.
                    # this piece of code was a test that failed miserably.
                    # whenever the agent reaches a state that contains a fact
                    # of the goal, give a positive reward.
                    #
                    # else:
                    #     if self.check_partial_goals:
                    #         r += check_for_partial_goals(obs, self.goal_literals_achieved)
                except InvalidAction:
                    next_state = state
                    reward = -1.
                    done = False

                if done:
                    done_times += 1

                # standard q-learning algorithm

                next_max_q = self.get_max_q(next_state)
                old_q = self.get_q_value(state, action)

                td_error = self.gamma*next_max_q - old_q
                new_q = old_q + self.alpha * (reward + td_error)

                self.set_q_value(state, action, new_q)
                state = next_state
                tstep += 1
                episode_r += reward
            if done:  # One last update at the terminal state
                old_q = self.get_q_value(state, action)

                td_error = - old_q

                new_q = old_q + self.alpha * (reward + td_error)
                # self.set_q_value(state, action, new_q)

            if episode_r > max_r:
                max_r = episode_r
                print("New all time high reward:", episode_r)
            if (n + 1) % 1000 == 0:
                print(f'Episode {n+1} finished. Timestep: {tstep}. Number of states: {len(self.q_table.keys())}. Reached the goal {done_times} times during this interval. Eps = {eps}')
                if done_times <= 10:
                    patience += 1
                    if patience >= self.patience:
                        print(f"Did not find goal after {n} episodes. Retrying.")
                        raise InvalidAction("Did not learn")
                else:
                    patience = 0
                if done_times == 1000 and converged_at is not None:
                    converged_at = n
                    print(f"***Policy converged to goal at {converged_at}***")
                done_times = 0
            self.goal_literals_achieved.clear()


class TabularDynaQLearner(TabularQLearner):
    def __init__(self,
                 env: Env,
                 init_obs: Any,
                 problem: int = None,
                 episodes: int = 30000,
                 decaying_eps: bool = True,
                 eps: float = 1.0,
                 alpha: float = 0.5,
                 decay: float = 0.000002,
                 gamma: float = 0.9,
                 action_list: Collection[Literal] = None,
                 rand: Random = Random(),
                 check_partial_goals: bool = True,
                 valid_only: bool = False,
                 planning_steps: int = 10,
                 **kwargs):
        self.planning_steps = planning_steps
        self.model = {}  # model is a dictionary of dictionaries, which maps states to actions to (reward, next_state) tuples

        super().__init__(env, init_obs, problem=problem, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, action_list=action_list, rand=rand, check_partial_goals=check_partial_goals, valid_only=valid_only, **kwargs)

    def update_model(self, past_state, past_action, state, reward):
        if past_state not in self.model:
            self.model[past_state] = {}
        self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        for i in range(self.planning_steps):
            past_state = self._random.choice(list(self.model.keys()))
            past_action = self._random.choice(list(self.model[past_state].keys()))
            state, reward = self.model[past_state][past_action]
            if state is None:
                td_error = - self.get_q_value(past_state, past_action)
            else:
                td_error = self.gamma*self.get_max_q(state) - self.get_q_value(past_state, past_action)
            new_q = self.get_q_value(past_state, past_action) + self.alpha*(reward + td_error)
            self.set_q_value(past_state, past_action, new_q)

    def learn(self):
        log_file = f'logs/tabular_q_learning_{datetime.datetime.now()}'
        tsteps = 50
        done_times = 0
        patience = 0
        converged_at = None
        max_r = float("-inf")
        init, _ = self.env.reset()
        print('LEARNING FOR GOAL:', init.goal)
        for n in range(self.episodes):
            episode_r = 0
            state, info = self.env.reset()
            state = solve_fset(state.literals)
            done = False
            tstep = 0
            while tstep < tsteps and not done:
                eps = self.eps()
                if self._random.random() <= eps:
                    action = self._random.randint(0, self.actions-1)
                    # a = self.env.action_space.sample(state)
                else:
                    action = self.best_action(state)
                try:
                    obs, reward, done, _ = self.env.step(self.action_list[action])
                    next_state = solve_fset(obs.literals)
                    if done:
                        reward = 100.
                    # this piece of code was a test that failed miserably.
                    # whenever the agent reaches a state that contains a fact
                    # of the goal, give a positive reward.
                    #
                    # else:
                    #     if self.check_partial_goals:
                    #         r += check_for_partial_goals(obs, self.goal_literals_achieved)
                except InvalidAction:
                    next_state = state
                    reward = -1.
                    done = False

                if done:
                    done_times += 1

                # Dyna q-learning algorithm

                next_max_q = self.get_max_q(next_state)
                old_q = self.get_q_value(state, action)

                td_error = self.gamma*next_max_q - old_q
                new_q = old_q + self.alpha * (reward + td_error)

                self.set_q_value(state, action, new_q)

                self.update_model(state, action, next_state, reward)
                self.planning_step()

                state = next_state
                tstep += 1
                episode_r += reward
            if done:  # One last update at the terminal state
                old_q = self.get_q_value(state, action)

                td_error = - old_q

                new_q = old_q + self.alpha * (reward + td_error)
                self.set_q_value(state, action, new_q)
                self.update_model(state, action, None, reward)
            if episode_r > max_r:
                max_r = episode_r
                print("New all time high reward:", episode_r)
            if (n + 1) % 1000 == 0:
                print(f'Episode {n+1} finished. Timestep: {tstep}. Number of states: {len(self.q_table.keys())}. Reached the goal {done_times} times during this interval. Eps = {eps}')
                if done_times <= 10:
                    patience += 1
                    if patience >= self.patience:
                        print(f"Did not find goal after {n} episodes. Retrying.")
                        raise InvalidAction("Did not learn")
                else:
                    patience = 0
                if done_times == 1000 and converged_at is not None:
                    converged_at = n
                    print(f"***Policy converged to goal at {converged_at}***")
                done_times = 0
            self.goal_literals_achieved.clear()
