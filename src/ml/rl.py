from abc import abstractmethod
import pickle
from typing import Any, Collection, List, NoReturn, overload
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
from tqdm import tqdm
from math import log2, exp
from queue import PriorityQueue

# This are for typing (we may want to move this elsewhere)
State = Any

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


def softmax(values: List[float]) -> List[float]:
    """Computes softmax probabilities for an array of values
    TODO We should probably use numpy arrays here
    Args:
        values (np.array): Input values for which to compute softmax

    Returns:
        np.array: softmax probabilities
    """
    return [(exp(q))/sum([exp(_q) for _q in values]) for q in values]


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
    def agent_start(self, state: State) -> Any:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        pass

    @abstractmethod
    def agent_step(self, reward: float, state: State) -> Any:
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
    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        pass

    @abstractmethod
    def policy(self, state: State) -> Any:
        """The action for the specified state under the currently learned policy
           (unlike agent_step, this does not update the policy using state as a sample.
           Args:
                state (Any): the state observation from the environment
           Returns:
                The action prescribed for that state
        """
        pass

    @abstractmethod
    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        pass

    @abstractmethod
    def learn(self, forced_init: bool = True, init_threshold: int = 20):
        pass

    def __getitem__(self, state: State) -> Any:
        """[summary]

        Args:
            state (Any): The state for which we want to get the policy

        Raises:
            InvalidAction: [description]

        Returns:
            Any: [description]
        """""
        return self.softmax_policy(state)


def print_q_values(q_values: Collection[int], actions: Collection[Literal]):
    values = ""
    for i, q in enumerate(q_values):
        values += f"{actions[i]}: {q}\n"
    return values


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
                if self.step == 0:
                    self.c_eps = max(self.c_eps - self.decay, 0.1)
                else:
                    self.c_eps = max((self.episodes - self.step)/self.episodes, 0.01)

                return self.c_eps
            self.eps = epsilon
        else:
            self.eps = lambda: eps
        self.decaying_eps = decaying_eps
        self.alpha = alpha
        self.last_state = None
        self.last_action = None

    def states_in_q(self) -> List:
        """Returns the states stored in the q_values table

        Returns:
            List: The states for which we have a mapping in the q-table
        """
        return self.q_table.keys()

    def policy(self, state: State) -> Any:
        """Returns the greedy deterministic policy for the specified state

        Args:
            state (State): the state for which we want the action

        Raises:
            InvalidAction: Not sure about this one

        Returns:
            Any: The greedy action learned for state
        """
        return self.best_action(state)

    def epsilon_greedy_policy(self, state: State) -> Any:
        eps = self.eps()
        if self._random.random() <= eps:
            action = self._random.randint(0, self.actions-1)
        else:
            action = self.policy(state)
        return action

    def softmax_policy(self, state: State) -> np.array:
        """Returns a softmax policy over the q-value returns stored in the q-table

        Args:
            state (State): the state for which we want a softmax policy

        Returns:
            np.array: probability of taking each action in self.actions given a state
        """
        if state not in self.q_table:
            self.add_new_state(state)
            # If we query a state we have not visited, return a uniform distribution
            # return softmax([0]*self.actions)
        return softmax(self.q_table[state])

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

    def add_new_state(self, state: State):
        # self.q_table[state] = [1. for _ in range(self.actions)]
        self.q_table[state] = [0.]*self.actions

    def get_all_q_values(self, state: State) -> List[float]:
        if state in self.q_table:
            return self.q_table[state]
        else:
            return [0.]*self.actions

    def best_action(self, state: State) -> int:
        if state not in self.q_table:
            self.add_new_state(state)
            # self.q_table[state] = [0 for _ in range(self.actions)]
        return np.argmax(self.q_table[state])

    def get_max_q(self, state: State) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return np.max(self.q_table[state])

    def set_q_value(self, state: State, action: Any, q_value: float):
        if state not in self.q_table:
            self.add_new_state(state)
        self.q_table[state][action] = q_value

    def get_q_value(self, state: State, action: Any) -> float:
        if state not in self.q_table:
            self.add_new_state(state)
        return self.q_table[state][action]

    def agent_start(self, state: State) -> int:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: State) -> int:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)

    def plan_optimistic_initialization(self, init: Any) -> NoReturn:
        planner = FD()
        init, _ = self.env.reset()
        plan = planner(self.env.domain, init)
        print(f'Initializing rewards using {plan}')
        obs, _ = self.env.reset()
        for pstep in plan:
            state = solve_fset(obs.literals)
            action = self.action_list.index(pstep)
            self.set_q_value(state, action, 100)  # I'm putting 1 as the reward not to bias this too much
            obs, reward, done, _ = self.env.step(self.action_list[action])
        assert(done)

    def learn(self, forced_init: bool = True, init_threshold: int = 20):
        log_file = f'logs/tabular_q_learning_{datetime.datetime.now()}'
        tsteps = 50
        done_times = 0
        patience = 0
        converged_at = None
        max_r = float("-inf")
        init, _ = self.env.reset()
        # planner = FD()
        # plan = planner(self.env.domain, init)
        # print(len(plan))
        if forced_init:
            self.plan_optimistic_initialization(init)
        print('LEARNING FOR GOAL:', init.goal)
        print(f'Using {self.__class__.__name__}')
        tq = tqdm(range(self.episodes), postfix=f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
        for n in tq:
            self.step = n
            episode_r = 0
            state, info = self.env.reset()
            state = solve_fset(state.literals)
            action = self.agent_start(state)
            done = False
            tstep = 0
            while tstep < tsteps and not done:
                # if forced_init and n < init_threshold and tstep < (len(plan)):
                #     action = self.action_list.index(plan[tstep])
                #     self.last_action = action
                #     print('Forced step:', action, tstep)
                try:
                    obs, reward, done, _ = self.env.step(self.action_list[action])
                    next_state = solve_fset(obs.literals)
                    if done:
                        reward = 100.
                except InvalidAction:
                    next_state = state
                    reward = -1.
                    done = False

                if done:
                    # print(".", end="")
                    done_times += 1

                # standard q-learning algorithm

                action = self.agent_step(reward, next_state)
                tstep += 1
                episode_r += reward
            if done:  # One last update at the terminal state
                self.agent_end(reward)

            if episode_r > max_r:
                max_r = episode_r
                # print("New all time high reward:", episode_r)
                tq.set_postfix_str(f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
            if (n + 1) % 1000 == 0:
                tq.set_postfix_str(f"States: {len(self.q_table.keys())}. Goals: {done_times}. Eps: {self.c_eps:.3f}. MaxR: {max_r}")
                # print(f'Episode {n+1} finished. Timestep: {tstep}. Number of states: {len(self.q_table.keys())}. Reached the goal {done_times} times during this interval. Eps = {self.c_eps}')
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


def run_training(agent: RLAgent):
    """A helper function to run agent training when trying to train multiple agents in parallel"""
    agent.learn()


class TabularDynaQLearner(TabularQLearner):
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
                 planning_steps: int = 10,
                 **kwargs):
        self.planning_steps = planning_steps
        self.model = {}  # model is a dictionary of dictionaries, which maps states to actions to (reward, next_state) tuples

        super().__init__(env, init_obs, problem=problem, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, action_list=action_list, rand=rand, check_partial_goals=check_partial_goals, valid_only=valid_only, **kwargs)

    def update_model(self, past_state: State, past_action, state: State, reward: float):
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

    def agent_start(self, state: State) -> Any:
        """The first method called when the experiment starts,
        called after the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            (int) the first action the agent takes.
        """
        self.last_state = state
        self.last_action = self.policy(state)
        return self.last_action

    def agent_step(self, reward: float, state: State) -> Any:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = self.gamma*max_q - old_q
        new_q = old_q + self.alpha * (reward + td_error)

        self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        action = self.epsilon_greedy_policy(state)

        self.update_model(self.last_state, self.last_action, state, reward)
        self.planning_step()

        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward: float) -> Any:
        """Called when the agent terminates.

        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        old_q = self.get_q_value(self.last_state, self.last_action)

        td_error = - old_q

        new_q = old_q + self.alpha * (reward + td_error)
        self.set_q_value(self.last_state, self.last_action, new_q)
        self.update_model(self.last_state, self.last_action, None, reward)


class TabularPrioritisedQLearner(TabularDynaQLearner):

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
                 planning_steps: int = 10,
                 priority_theta: float = 85,  # TODO Check that this is agood threshold
                 **kwargs):
        self.pqueue = PriorityQueue()
        self.priority_theta = priority_theta
        self.inverse_model = {}  # inverse_model stores state-actions that lead to a state

        super().__init__(env, init_obs, problem=problem, episodes=episodes, decaying_eps=decaying_eps, eps=eps, alpha=alpha, decay=decay, gamma=gamma, action_list=action_list, rand=rand, check_partial_goals=check_partial_goals, valid_only=valid_only, planning_steps=planning_steps, **kwargs)

    def update_model(self, past_state: State, past_action, state: State, reward: float):
        if past_state not in self.model:
            self.model[past_state] = {}
        if state not in self.inverse_model:
            self.inverse_model[state] = []
        self.inverse_model[state].append((past_state, past_action, reward))
        self.model[past_state][past_action] = (state, reward)

    def planning_step(self):
        for i in range(self.planning_steps):
            if self.pqueue.empty():
                return

            _, past_state, past_action = self.pqueue.get()
            state, reward = self.model[past_state][past_action]
            if state is None:
                td_error = - self.get_q_value(past_state, past_action)
            else:
                td_error = self.gamma*self.get_max_q(state) - self.get_q_value(past_state, past_action)
            new_q = self.get_q_value(past_state, past_action) + self.alpha*(reward + td_error)
            self.set_q_value(past_state, past_action, new_q)
            for back_state, back_action, back_reward in self.inverse_model[past_state]:
                priority = abs(back_reward + self.gamma*self.get_max_q(past_state) - self.get_q_value(back_state, back_action))
                if priority > self.priority_theta:
                    self.pqueue.put((priority, back_state, back_action))
        # while not self.pqueue.empty():  # Empty out the queue at the end of the planning steps
        #     self.pqueue.get()
        # self.pqueue.task_done()  # This should empty out the queue (hopefully)
        self.pqueue = PriorityQueue()

    def agent_step(self, reward: float, state: State) -> Any:
        """A step taken by the agent.

        Args:
            reward (float): the reward received for taking the last action taken
            state (Any): the state from the
                environment's step based on where the agent ended up after the
                last step
        Returns:
            (int) The action the agent takes given this state.
        """
        max_q = self.get_max_q(state)
        old_q = self.get_q_value(self.last_state, self.last_action)
        self.update_model(self.last_state, self.last_action, state, reward)

        td_error = self.gamma*max_q - old_q
        # new_q = old_q + self.alpha * (reward + td_error)
        # self.set_q_value(self.last_state, self.last_action, new_q)
        # action = self.best_action(state)
        priority = abs(reward + td_error)
        if priority > self.priority_theta:
            self.pqueue.put((priority, self.last_state, self.last_action))

        self.planning_step()

        action = self.epsilon_greedy_policy(state)

        self.last_state = state
        self.last_action = action
        return action
