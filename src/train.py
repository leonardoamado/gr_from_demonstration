from numpy import deg2rad
from ml.rl import TabularQLearner
import json
import time
# from ..env_manager import EnvManager
from matplotlib import pyplot as plt
import ml.metrics as m
import sys, os
sys.path.append(os.path.abspath(os.path.join('..')))

import numpy as np
from pddlgym_planners.fd import FD

import pddlgym

env = pddlgym.make("PDDLEnvBlocks-v0", raise_error_on_invalid_action=False)

train = True
policies = []
n_goals = 4

actions = None

for n in range(n_goals):
    time.sleep(3)
    env.fix_problem_index(n)
    obs, _ = env.reset()
    if not actions:
        actions = list(env.action_space.all_ground_literals(obs, valid_only=False))
    policy = TabularQLearner(env, obs, problem=n, action_list=actions)
    policies.append(policy)
    if train:
        done = False
        print(f"Training policy for goal {n}")
        while not done:
            try:
                policy.learn()
                done = True
            except ValueError:
                obs, _ = env.reset()
                policies[-1] = TabularQLearner(env, obs, problem=n, action_list=actions)
                policy = policies[-1]
        # policy.save_q_table(f"policies/p{n}.pickle")
    else:
        policy.load_q_table(f"policies/p{n}.pickle")

planner = FD()

epsilons = [0., 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45]

for n in range(n_goals):
    env.fix_problem_index(n)
    obs, _ = env.reset()
    print(f"PLANNING FOR GOAL {obs.goal}")
    plan = planner(env.domain, obs)

    # traj is an action pair tuple, need to map this to state action number pair
    traj = []
    for a in plan:
        state_action_pair = (obs.literals, a)
        traj.append(state_action_pair)
        obs, _, _, _ = env.step(a)
    ds = {}
    for eps in epsilons:
        d1 = m.kl_divergence_norm(traj, policies[0].q_table, actions, epsilon=eps)
        d2 = m.kl_divergence_norm(traj, policies[1].q_table, actions, epsilon=eps)
        d3 = m.kl_divergence_norm(traj, policies[2].q_table, actions, epsilon=eps)
        d4 = m.kl_divergence_norm(traj, policies[3].q_table, actions, epsilon=eps)
        if f'p{n}' not in ds:
            ds[f'p{n}'] = []
        ds[f'p{n}'].append([d1, d2, d3, d4])
        # m.plot_traj_policy_divergence(n, eps, d1)
        m.plot_mean_divergence(n, eps, d1, d2, d3, d4)

    # m.plot(n, d1, d2, d3)


# print(obs)1
# for key in learner.q_table.keys():
#     print(learner.q_table[key])
# while not done:
#     a = learner.best_action(obs)
#     a = learner.action_list[a]
#     print(a)
#     obs, _, done, _ = env.step(a)


#     pickup(b:block)
    # stack(b:block,a:block)
    # pickup(c:block)
    # stack(c:block,b:block)
    # pickup(d:block)
    # stack(d:block,c:block)