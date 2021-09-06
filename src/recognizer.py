from types import MethodType
from typing import Any, Collection, List, Tuple
from gym.core import Env
from numpy import deg2rad
from ml.rl import RLAgent, TabularQLearner
from ml.dqn import DQN
import json
import time
from matplotlib import pyplot as plt
import ml.metrics as metrics
import sys
import os
import dill
import numpy as np
import argparse
from pddlgym.core import InvalidAction, PDDLEnv
from utils import solve_fset, rebuild_qtable
import pddlgym
from pddlgym.core import InvalidAction, PDDLEnv
from pddlgym.structs import Literal
import random
import pickle
# import multiprocessing as mp  # To train policies in parallel
from multiprocessing import Pool
# from pathos.pools import ParallelPool as Pool
import copy
sys.path.append(os.path.abspath(os.path.join('..')))
from pddlgym_planners.fd import FD

"""
This is a copy of the train.py file with a few adjustments to Gabriel's code.
This file will become the main file for the goal recognition process, and is
still a WIP.
"""

# configs for environment
RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True

# when creating an env, if raise_error_on_invalid_action is true,
# whenever an invalid action is performed, an InvalidAction error is raised.
# dynamic_action_space makes the env return only the valid actions of a state
# when listing possible actions.


#############################


# current_method = DQN
# This function will be moved to another file, leave this here for now
def remove_obs(instance, observability):
    new_obs = []
    n_observations = len(instance)

    # Number of observations to remove
    n_remove = int(n_observations*(1-observability))

    # Randomly sample indices to remove from the states list
    indices = sorted(random.sample(range(0, n_observations), n_remove))

    # Create new list with states except the indices to remove
    for i in range(n_observations):
        if i not in indices:
            new_obs.append(instance[i])
    return new_obs


class Recognizer:

    def __init__(self, method: RLAgent = TabularQLearner,
                 evaluation: MethodType = metrics.kl_divergence_norm_softmax  # This does not seem to be used
                 ):
        """Initializes a recognizer

        Args:
            method (RLAgent, optional): Which RL method we are using to train the agents (and derive policies). Defaults to TabularQLearner.
            evaluation (MethodType, optional): The actual method to evaluate a policy against a trajectory. Note the signature must be: evaluation(traj: List[Tuple[State, Any]], pol: RLAgent, actions: List[Literal])
            . Defaults to m.kl_divergence_norm_softmax
        """
        self.method = method
        self.evaluate_goal = evaluation

    '''
    Performs the entire process of goal recognition using the user assigned functions.
    @return the predicted goal
    '''
    def complete_recognition(self, env, n_goals=3, obs=None, real_goal=None):
        policies, actions = self.train_policies(env, n_goals)
        goal = self.recognize_process(env, policies, actions, obs, real_goal)
        return goal

    def load_correct_goal(self, file):
        correct_goal = 0
        with open(file, 'rb') as goal:
            correct_goal = int(goal.readline())
        return correct_goal

    '''
    TODO Refactor this method away from this class, as it does not belong here
    Performs the entire process of goal recognition using the user a folder as parameter.
    The recognizer starts with the default conditions
    @return the predicted goal
    '''
    def complete_recognition_folder(self, folder, observations=[0.1, 0.3, 0.5, 0.7, 1.0]):
        print('Recognizing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
        env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                      dynamic_action_space=DYNAMIC_ACTION_SPACE)
        obs_traces = []
        n_goals = len(env.problems)
        real_goal = self.load_correct_goal(folder + 'real_hypn.dat')
        for obs in observations:
            with open(folder + 'obs' + str(obs) + '.pkl', "rb") as input_file:
                obs_traces.append(pickle.load(input_file))

        policies, actions = self.train_policies(env, n_goals)
        with open(folder + 'policies.pkl', 'wb') as file:
            dill.dump(policies, file)
        with open(folder + 'actions.pkl', 'wb') as file:
            dill.dump(actions, file)
        result_list = []
        for trace in obs_traces:
            correct, goal, rankings = self.recognize_process(env, policies, actions, tuple(trace), real_goal, n_goals)
            result_list.append((correct, goal, rankings))
        return result_list

    def only_recognition_folder(self, folder, observations=[0.1, 0.3, 0.5, 0.7, 1.0]):
        print('Recognizing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
        env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/', raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                      dynamic_action_space=DYNAMIC_ACTION_SPACE)
        obs_traces = []
        n_goals = len(env.problems)
        real_goal = self.load_correct_goal(folder + 'real_hypn.dat')
        for obs in observations:
            with open(folder + 'obs' + str(obs) + '.pkl', "rb") as input_file:
                obs_traces.append(pickle.load(input_file))

        with open(folder + 'policies.pkl', 'rb') as file:
            policies = dill.load(file)
        with open(folder + 'actions.pkl', 'rb') as file:
            actions = dill.load(file)
        for policy in policies:
            policy.q_table = rebuild_qtable(policy.q_table)
        result_list = []
        for trace in obs_traces:
            correct, goal, rankings = self.recognize_process(env, policies, actions, tuple(trace), real_goal, n_goals)
            result_list.append((correct, goal, rankings))
        return result_list

    ''' TODO FRM: Justify this or remove altogether
    Performs a dummy goal recognition process, where observations are not considered.

    The first goal is considered the correct one and a plan is computed for the goal
    which serves as a full observation trace.
    @return a goal
    '''
    def recognize_goal_dummy(self, env: Env, policies: Collection[RLAgent], actions: RLAgent, obs: List, real_goal: int, n_goals: int = 3) -> int:
        planner = FD()
        traj = []
        divergences = []
        list_of_goals = []

        for n in range(n_goals):
            env.fix_problem_index(n)
            init, _ = env.reset()
            list_of_goals.append(init.goal)

            # traj is an action pair tuple, need to map this to state action number pair
            if len(traj) == 0:
                print(f"Dummy recog, PLANNING FOR GOAL {init.goal}")
                plan = planner(env.domain, init)
                traj = []
                for a in plan:
                    state_action_pair = (solve_fset(init.literals), a)
                    traj.append(state_action_pair)
                    init, _, _, _ = env.step(a)
                # Forcefully remove observations, just for testing
                # traj = remove_obs(traj, 0.5)
            divergence = self.evaluate_goal(traj, policies[n], actions)
            divergences.append(divergence)
        print(divergences)
        div, goal = min((div, goal) for (goal, div) in enumerate(divergences))
        print(f'Most likely goal is: {goal} with metric value {self.evaluate_goal}: {div}')

        return goal

    '''
    Performs the goal recognition process, where observations are considered.

    Arguments:
        env -- a PDDLGym env
        policies -- a list of policies (actually RLAgents)
        actions -- possible actions of the environment (env.action_space.all_ground_literals(init, valid_only=False))
        obs -- it is expected to be a list of state-action pairs
        real_goal -- it is spected to be the index of the correct goal
        n_goals -- the number of goals

    @return a goal
    '''
    def recognize_process(self, env: Env, policies: Collection[RLAgent], actions: Collection[Literal], obs: Collection, real_goal: int, n_goals: int = 3) -> Tuple[bool, int, List[Tuple[int, Any]]]:
        divergences = []
        list_of_goals = []
        for tup1 in obs:  # TODO This is a hack due to PDDLGym's initialization
            for pred in tup1[0]:
                pred._hash = hash(pred._str)
            tup1[1]._hash = hash(tup1[1]._str)
        for n in range(n_goals):
            env.fix_problem_index(n)
            init, _ = env.reset()
            list_of_goals.append(init.goal)
            divergence = self.evaluate_goal(obs, policies[n], actions)
            divergences.append(divergence)
        print(divergences)
        rankings = sorted(((goal, div) for (goal, div) in enumerate(divergences)), key=lambda tup: tup[0])
        div, goal = min((div, goal) for (goal, div) in enumerate(divergences))
        print(f'Most likely goal is: {goal} with metric value {self.evaluate_goal}: {div}')
        print('Correct prediction:', goal == real_goal)
        return goal == real_goal, goal, rankings

    def train_policies(self, env: Env, n_goals: int = 3) -> Tuple[List[RLAgent], Collection[Literal]]:
        '''
        Train a policy for each one of the goals.
        @return a list of policies and and the possible actions of the environment
        '''
        policies = []
        actions = None
        starting_problem_index = 0
        for n in range(n_goals):
            time.sleep(3)
            env.fix_problem_index(starting_problem_index + n)
            init, _ = env.reset()
            if not actions:
                # get all literals in its grounded version
                actions = list(env.action_space.all_ground_literals(init, valid_only=False))
            # build method to learn policy
            policy: RLAgent = self.method(env, init, problem=n, action_list=actions, valid_only=DYNAMIC_ACTION_SPACE)
            policies.append(policy)
            done = False
            print(f"Training policy for goal {n}")
            while not done:
                try:
                    policy.learn()
                    done = True
                except ValueError as e:
                    print(e)
                    init, _ = env.reset()
                    policies[-1] = self.method(env, init, problem=n, action_list=actions)
                    policy = policies[-1]
        return policies, actions

    def train_parallel(self, env: Env, n_goals: int = 3) -> Tuple[List[RLAgent], Collection[Literal]]:
        actions = None
        starting_problem_index = 0
        envs: List[Env] = [copy.deepcopy(env) for i in range(n_goals)]
        for n in range(n_goals):
            envs[n].fix_problem_index(n)
        policies: List[RLAgent] = [self.method(env, env.reset()[0], problem=n, action_list=actions, valid_only=DYNAMIC_ACTION_SPACE) for n, env in enumerate(envs)]
        print(f"Training policies in parallel for {n_goals} goals")
        pool = Pool(n_goals)
        # pool.map(lambda policy: policy.learn(), policies)
        pool.map(run_training, policies)
        # processes = [mp.Process(target=policy.learn) for policy in policies]
        # for process in processes:
        #     print("Starting process")
        #     process.start()
        # print("Started all processes")
        # for process in processes:
        #     process.join()
        print(f"Finished training {n_goals} policies")

        return policies, actions

    def __repr__(self) -> str:
        return f"Recognizer: {self.__class__.__name__} -- Evaluation: {self.evaluate_goal.__name__}"


class StateQmaxRecognizer(Recognizer):
    """A Goal Recognition Process that uses only states as observations"""
    def __init__(self, method: RLAgent = TabularQLearner):
        super().__init__(method=method, evaluation=self.evaluate_goal)

    def recognize_process(self, env: Env, policies: RLAgent, actions: Collection[Literal], obs: Collection, real_goal: int, n_goals: int) -> Tuple[bool, int, List[Tuple[int, Any]]]:
        observation_Qs = []
        list_of_goals = []
        for tup1 in obs:  # TODO This is a hack due to PDDLGym's initialization
            for pred in tup1[0]:
                pred._hash = hash(pred._str)
            tup1[1]._hash = hash(tup1[1]._str)
        for n in range(n_goals):
            env.fix_problem_index(n)
            init, _ = env.reset()
            list_of_goals.append(init.goal)
        # for n in range(n_goals):
        #     env.fix_problem_index(n)
        #     init, _ = env.reset()
        #     list_of_goals.append(init.goal)
        #     observation_q = self.evaluate_goal(obs, policies[n], actions)
        #     observation_Qs.append(observation_q)
        for state, _ in obs:
            stateQs = [policy.get_max_q(state) for policy in policies]
            stateQs = stateQs/np.max(stateQs)  # Normalize stateQs
            # stateQs = np.where(stateQs == np.max(stateQs), 1, 0)  # Choose max values
            observation_Qs.append(stateQs)
        print(observation_Qs)
        observation_Qs = np.sum(observation_Qs, axis=0)
        print(observation_Qs)
        rankings = sorted(((goal, div) for (goal, div) in enumerate(observation_Qs)), key=lambda tup: tup[0], reverse=True)
        div, goal = max((div, goal) for (goal, div) in enumerate(observation_Qs))
        print(f'Most likely goal is: {goal} with metric value {self.evaluate_goal}: {div}')
        print('Correct prediction:', goal == real_goal)
        return goal == real_goal, goal, rankings

    def evaluate_goal(self, obs: List[Tuple], policy: RLAgent, actions: Collection[Literal]) -> float:
        obs_q = 0
        for state, _ in obs:
            obs_q += policy.get_max_q(state)
        return obs_q


class ActionQmaxRecognizer(Recognizer):

    """A Goal Recognition Process that uses only actions as observations"""
    def __init__(self, method: RLAgent = TabularQLearner):
        super().__init__(method=method, evaluation=self.evaluate_goal)

    def recognize_process(self, env: Env, policies: RLAgent, actions: Collection[Literal], obs: Collection, real_goal: int, n_goals: int) -> Tuple[bool, int, List[Tuple[int, Any]]]:
        observation_Qs = []
        list_of_goals = []
        for tup1 in obs:  # TODO This is a hack due to PDDLGym's initialization
            for pred in tup1[0]:
                pred._hash = hash(pred._str)
            tup1[1]._hash = hash(tup1[1]._str)
        for n in range(n_goals):
            env.fix_problem_index(n)
            init, _ = env.reset()
            list_of_goals.append(init.goal)
        # for n in range(n_goals):
        #     env.fix_problem_index(n)
        #     init, _ = env.reset()
        #     list_of_goals.append(init.goal)
        #     observation_q = self.evaluate_goal(obs, policies[n], actions)
        #     observation_Qs.append(observation_q)
        for _, action in obs:
            action_index = actions.index(action)
            statesForAction = {}
            for policy in policies:
                statesForAction[policy] = [state for state in policy.q_table.keys()
                                           if policy.policy(state) == action_index]
                # print(f"States for action: {len(statesForAction[policy])}")
                # print(f"Q-values: {[policy.get_max_q(state) for state in statesForAction[policy]]}")

            # stateQs = [np.average([policy.get_max_q(state) for state in statesForAction[policy]]) for policy in policies]
            stateMaxQs = {policy: [0]+([policy.get_max_q(state) for state in statesForAction[policy]]) for policy in policies}  # We need to have a zero here to ensure we have something in case there are no states
            # stateMaxQs = {}
            # for policy in policies:
            #     if statesForAction[policy]:
            #         stateMaxQs[policy] = [0]
            #     else:
            #         stateMaxQs[policy] = [policy.get_max_q(state) for state in statesForAction[policy]]
            stateQs = [np.max(stateMaxQs[policy]) for policy in policies]
            # stateQs = [np.max([policy.get_max_q(state) for state in statesForAction[policy]]) for policy in policies]
            stateQs = stateQs/np.max(stateQs)  # Normalize stateQs
            # stateQs = np.where(stateQs == np.max(stateQs), 1, 0)  # Choose max values
            observation_Qs.append(stateQs)
        print(observation_Qs)
        observation_Qs = np.sum(observation_Qs, axis=0)
        print(observation_Qs)
        rankings = sorted(((goal, div) for (goal, div) in enumerate(observation_Qs)), key=lambda tup: tup[0], reverse=True)
        div, goal = max((div, goal) for (goal, div) in enumerate(observation_Qs))
        print(f'Most likely goal is: {goal} with metric value {self.evaluate_goal}: {div}')
        print('Correct prediction:', goal == real_goal)
        return goal == real_goal, goal, rankings

    def evaluate_goal(self, obs: List[Tuple], policy: RLAgent, actions: Collection[Literal]) -> float:
        obs_q = 0
        for state, _ in obs:
            obs_q += policy.get_max_q(state)
        return obs_q


def str_to_literal(string):
    '''
    Reads the string representation of a PDDLGym literal and returns a PDDLGym literal
    Literal definition : https://github.com/tomsilver/pddlgym/blob/3f1df3bff0e50dea36570df926703fd46977d246/pddlgym/structs.py#L186
    '''
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", help="PDDL domain file",
                        type=str, default='dummy')
    parser.add_argument("-p", help="Folder with all problems",
                        type=str)
    parser.add_argument("-t", help="Computing policies or loading and recognizing",
                        type=str, default='learn')
    parser.add_argument("-pl", help="Load or save policies file",
                        type=str, default='policies_saved.pkl')
    parser.add_argument("-a", help="Load or save actions file",
                        type=str, default='saved.pkl')
    args = parser.parse_args()
    print(args.d)
    if args.d == 'dummy':
        # env = pddlgym.make("PDDLEnvBlocks_gr-v0",
        #                     raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
        #                     dynamic_action_space=DYNAMIC_ACTION_SPACE)
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=RAISE_ERROR_ON_VALID, dynamic_action_space=DYNAMIC_ACTION_SPACE)
    else:
        env = PDDLEnv(args.d, args.p)
    recog = Recognizer()
    if args.t == 'complete':
        # TODO Load observations and goal
        recog.complete_recognition(env)
    if args.t == 'learn':
        policies, actions = recog.train_policies(env)
        with open(args.pl, 'wb') as file:
            dill.dump(policies, file)
        with open(args.a, 'wb') as file:
            dill.dump(actions, file)
    if args.t == 'load':
        if args.t == 'load':
            with open(args.pl, 'rb') as file:
                policies = dill.load(file)
            with open(args.a, 'rb') as file:
                actions = dill.load(file)
        n_goals = 3
        recog.recognize_goal_dummy(env, policies, actions, None, None)
    if args.t == 'foldern':
        recog = Recognizer(evaluation=metrics.soft_divergence_point)
        recog.complete_recognition_folder('output/blocks_gr/')
