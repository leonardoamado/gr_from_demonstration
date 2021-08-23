from numpy import deg2rad
from ml.linear import LinearQLearning
from ml.rl import TabularQLearner
from ml.dqn import DQN
import json
import time
# from ..env_manager import EnvManager
from matplotlib import pyplot as plt
import ml.metrics as m
import sys
import os
sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.abspath(os.path.join('..')))
import dill
import numpy as np
import argparse
from pddlgym_planners.fd import FD

from pddlgym.core import InvalidAction, PDDLEnv
#from pddlgym import PDDLEnv
import pddlgym
import random



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
            self.recognize_process = self.recognize_goal_dummy
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
    

    '''
    Performs a dummy goal recognition process, where observations are not considered.
    
    The first goal is considered the correct one and a plan is computed for the goal
    which serves as a full observation trace.
    @return a goal
    '''
    def recognize_goal_dummy(self, env, policies, actions, obs, real_goal, n_goals=3):
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
                    state_action_pair = (init.literals, a)
                    traj.append(state_action_pair)
                    init, _, _, _ = env.step(a)
                # Forcefully remove observations, just for testing    
                traj = remove_obs(traj, 0.5)
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
        policies -- a list of policies
        actions -- possible actions of the environment (env.action_space.all_ground_literals(init, valid_only=False))
        obs -- it is spected to be a list of state-action pairs
        real_goal -- it is spected to be the index of the correct goal
        n_goals -- the number of goals

    @return a goal
    '''
    def recognize_goal(self, env, policies, actions, obs, real_goal, n_goals=3):
        traj = obs
        divergences = []
        list_of_goals = []

        for n in range(n_goals):
            env.fix_problem_index(n)
            init, _ = env.reset()
            list_of_goals.append(init.goal)
            divergence = self.evaluate_goal(obs, policies[n], actions)
            divergences.append(divergence)
        print(divergences)
        div, goal = min((div, goal) for (goal, div) in enumerate(divergences))
        print('Most likely goal is:', goal, 'with metric value (standard is KL_divergence):', div)
        print('Correct goal is ', real_goal)
        return goal
    '''
    Train a policy for each one of the goals. 
    @return a list of policies and and the possible actions of the environment
    '''
    def train(self, env, n_goals=3):
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
    
    
