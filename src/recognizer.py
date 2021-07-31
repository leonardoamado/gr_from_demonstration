from numpy import deg2rad
from ml.linear import LinearQLearning
from ml.rl import TabularQLearner
from ml.dqn import DQN
import json
import time
# from ..env_manager import EnvManager
from matplotlib import pyplot as plt
import ml.metrics as m
import sys, os
sys.path.append(os.path.abspath(os.path.join('.')))
sys.path.append(os.path.abspath(os.path.join('..')))
import dill
import numpy as np
from pddlgym_planners.fd import FD

from pddlgym.core import InvalidAction

import pddlgym


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
def train_policies(env, method=TabularQLearner, n_goals=3):
    train = True
    policies = []
    robots = 1

    offsets = {}

    predicates = env.observation_space.predicates

    state_representation = []

    actions = None

    current_method = method

    starting_problem_index = 1

    for n in range(n_goals):
        time.sleep(3)
        env.fix_problem_index(starting_problem_index + n)
        init, _ = env.reset()
        if not actions:
            # get all literals in its grounded version
            actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        # policy = TabularQLearner(env, init, problem=n, action_list=actions)
        # build method to learn policy
        policy = current_method(env, init, problem=n, action_list=actions, valid_only=DYNAMIC_ACTION_SPACE)
        policies.append(policy)
        if train:
            done = False
            print(f"Training policy for goal {n}")
            while not done:
                try:
                    policy.learn()
                    done = True
                except ValueError as e:
                    # if this error happened, it means that the policy did not learn
                    # how to reach the goal after a large number of training steps.
                    # In this case, reset the learning and try again.
                    # (this was just a dumb fix to when the agent randomly didn't reach the goal
                    # before epsilon getting too low)
                    print(e)
                    init, _ = env.reset()
                    policies[-1] = current_method(env, init, problem=n, action_list=actions)
                    policy = policies[-1]
    return policies, actions


def plot_divergences(env, actions, n_goals=3):

    planner = FD()

    epsilons = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

    for n in range(n_goals):
        # for each goal, get an optimal plan and compute the
        # divergence of each policy to this trajectory.

        env.fix_problem_index(n)
        init, _ = env.reset()
        print(f"PLANNING FOR GOAL {init.goal}")
        plan = planner(env.domain, init)

        # traj is an action pair tuple, need to map this to state action number pair
        traj = []
        for a in plan:
            state_action_pair = (init.literals, a)
            traj.append(state_action_pair)
            init, _, _, _ = env.step(a)
        ds = {}
        
        # compute policy with multiple epsilon values (only relevant when not doing softmax)
        for eps in epsilons:
            
            # pretty bad implementation of a plot, but w.e.
            divergences = []
            for i in range(n_goals):
                divergence = m.kl_divergence_norm_softmax(traj, policies[i].q_table, actions, epsilon=eps)
                divergences.append(divergence)
            # d1 = m.kl_divergence_norm(traj, policies[0].q_table, actions, epsilon=eps)
            # d2 = m.kl_divergence_norm(traj, policies[1].q_table, actions, epsilon=eps)
            # d3 = m.kl_divergence_norm(traj, policies[2].q_table, actions, epsilon=eps)
            # d4 = m.kl_divergence_norm(traj, policies[3].q_table, actions, epsilon=eps)
            # d1 = m.kl_divergence_norm_softmax(traj, policies[0].q_table, actions, epsilon=eps)
            # d2 = m.kl_divergence_norm_softmax(traj, policies[1].q_table, actions, epsilon=eps)
            # d3 = m.kl_divergence_norm_softmax(traj, policies[2].q_table, actions, epsilon=eps)
            # d4 = m.kl_divergence_norm_softmax(traj, policies[3].q_table, actions, epsilon=eps)
            # if f'p{n}' not in ds:
            #     ds[f'p{n}'] = []
            # ds[f'p{n}'].append([d1, d2, d3, d4])
            # m.plot_mean_divergence(n, eps, d1, d2, d3, d4)
            m.plot_mean_divergence(n, eps, divergences)





if __name__ == "__main__":

    env = pddlgym.make("PDDLEnvBlocks_gr-v0",
                        raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                        dynamic_action_space=DYNAMIC_ACTION_SPACE)
    #env = PDDLEnv(“pddl/sokoban.pddl”, “pddl/sokoban”)
    policies, actions  = train_policies(env)
    with open('policies_saved.pkl', 'wb') as file:
        dill.dump(policies, file)
    with open('saved.pkl', 'wb') as file:
        dill.dump(actions, file)
    
    
