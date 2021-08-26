from typing import Any, Collection, List, Tuple
from gym.core import Env
from numpy import deg2rad
from ml.linear import LinearQLearning
from ml.rl import RLAgent, TabularQLearner
from ml.dqn import DQN
import json
import time
# from ..env_manager import EnvManager
from matplotlib import pyplot as plt
import ml.metrics as m
import sys
import os
import dill
import numpy as np
import argparse
from pddlgym.core import InvalidAction, PDDLEnv
from utils import solve_fset
import pddlgym
from pddlgym.core import InvalidAction, PDDLEnv
from pddlgym.structs import Literal
import random
import pickle
# sys.path.append(os.path.abspath(os.path.join('.')))
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
    def __init__(self, method=TabularQLearner, evaluation=m.kl_divergence_norm_softmax, training=None, recog=None):
        self.method = method
        self.evaluate_goal = evaluation
        if not training:
            self.train_policies = self.train
        else:
            self.train_policies = training
        if not recog:
            self.recognize_process = self.recognize_goal
        else:
            self.recognize_process = recog

    '''
    Performs the entire process of goal recognition using the user assigned functions.
    @return the predicted goal
    '''
    def complete_recognition(self, env, n_goals=3,obs=None, real_goal=None):
        policies, actions = self.train_policies(env, n_goals)
        goal = self.recognize_process(env, policies, actions, obs, real_goal)
        return goal

    def load_correct_goal(self, file):
        correct_goal = 0
        with open(file, 'rb') as goal:
            correct_goal = int(goal.readline())
        return correct_goal

    '''
    Performs the entire process of goal recognition using the user a folder as parameter.
    The recognizer starts with the default conditions
    @return the predicted goal
    '''
    def complete_recognition_folder(self, folder, observations=[0.1,0.3,0.5,0.7,1.0]):
        print('Recognizing domain:', folder + 'domain.pddl', 'problems:', folder + 'problems/')
        env = PDDLEnv(folder + 'domain.pddl', folder + 'problems/',raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                            dynamic_action_space=DYNAMIC_ACTION_SPACE)
        obs_traces = []
        n_goals = len(env.problems)
        real_goal = self.load_correct_goal(folder + 'real_hypn.dat')
        for obs in observations:
            with open(folder +'obs' + str(obs)+'.pkl', "rb") as input_file:
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



    ''' TODO FRM: Justify this or remove altogether
    Performs a dummy goal recognition process, where observations are not considered.

    The first goal is considered the correct one and a plan is computed for the goal
    which serves as a full observation trace.
    @return a goal
    '''
    def recognize_goal_dummy(self, env: Env, policies, actions: RLAgent, obs: List, real_goal: int, n_goals: int = 3) -> int:
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
                #Forcefully remove observations, just for testing    
                #traj = remove_obs(traj, 0.5)
            divergence = self.evaluate_goal(traj, policies[n], actions)
            divergences.append(divergence)
        print(divergences)
        div, goal = min((div, goal) for (goal, div) in enumerate(divergences))
        print('Most likely goal is:', goal, 'with metric value (standard is KL_divergence):', div)
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
    def recognize_goal(self, env: Env, policies: RLAgent, actions: Collection[Literal], obs: Collection, real_goal: int, n_goals: int = 3) -> Tuple[bool, int, List[Tuple[int, Any]]]:
        divergences = []
        list_of_goals = []
        for tup1 in obs:
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
        print('Most likely goal is:', goal, 'with metric value (standard is KL_divergence):', div)
        print('Correct prediction:', goal==real_goal)
        return goal==real_goal, goal, rankings


    def train(self, env: Env, n_goals: int = 3) -> Tuple[RLAgent, Collection[Literal]]:
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
            policy = self.method(env, init, problem=n, action_list=actions, valid_only=DYNAMIC_ACTION_SPACE)
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
        recog.recognize_goal_dummy(env,policies, actions, None, None)
    if args.t == 'foldern':
        recog = Recognizer()
        recog.complete_recognition_folder('output/blocks_gr/')
