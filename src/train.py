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

env = pddlgym.make("PDDLEnvBlocks-v0", raise_error_on_invalid_action=True)
# obs, _ = env.reset()
# env.fix_problem_index(0)
# learner = TabularQLearner(env, obs, problem=1)
# learner.learn()
# obs, _ = env.reset()
# done = False

train = True
policies = []

for n in range(3):
    time.sleep(3)
    env.fix_problem_index(n)
    obs, _ = env.reset()
    policy = TabularQLearner(env, obs, problem=n)
    policies.append(policy)
    if train:
        print(f"Training policy for goal {n}")
        policy.learn()
        # policy.save_q_table(f"policies/p{n}.pickle")
    else:
        policy.load_q_table(f"policies/p{n}.pickle")

planner = FD()


for n in range(3):
    env.fix_problem_index(n)
    obs, _ = env.reset()
    print(f"PLANNING FOR GOAL {obs.goal}")
    plan = planner(env.domain, obs)
    traj = [obs]
    for a in plan:
        obs, _, _, _ = env.step(a)
        traj.append(obs)
    d1 = m.kl_divergence_per_plan_state(traj, policies[n].q_table, policies[0].q_table)
    d2 = m.kl_divergence_per_plan_state(traj, policies[n].q_table, policies[1].q_table)
    d3 = m.kl_divergence_per_plan_state(traj, policies[n].q_table, policies[2].q_table)
    m.plot(n, d1, d2, d3)


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