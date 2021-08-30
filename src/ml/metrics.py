import collections
from typing import Collection
from matplotlib import pyplot as plt
import numpy as np
from math import log2, exp
import random
import os

from numpy.core.fromnumeric import mean


def kl_divergence(p1, p2):
    return sum(p1[i] * log2(p1[i]/p2[i]) for i in range(len(p1)))


def softmax(values):
    return [(exp(q))/sum([exp(_q) for _q in values]) for q in values]

# def generic_kl_divergence_per_plan_state(trajectory, policy, epsilon=0., actions=24):


def kl_divergence_per_plan_state(trajectory, p1, p2, epsilon=0., actions=24):
    p1 = values_to_policy(p1, epsilon)
    p2 = values_to_policy(p2, epsilon)

    per_state_divergence = []
    for state in trajectory:
        if state not in p1:
            p1[state] = [1e-6 + epsilon/actions for _ in range(actions)]
            random_best_action = random.choice(range(actions))
            p1[state][random_best_action] = 1. - 1e-6*(actions-1) - epsilon
        if state not in p2:
            p2[state] = [1e-6 + epsilon/actions for _ in range(actions)]
        qp1 = p1[state]
        qp2 = p2[state]
        per_state_divergence.append(kl_divergence(qp1, qp2))
    return per_state_divergence


def kl_divergence_norm_generic(traj, policy, actions, epsilon=0.):
    policy_trajectory = traj_to_policy(traj, actions)
    distances = []
    for i, state in enumerate(policy_trajectory):
        q_trajectory = policy_trajectory[state]
        q_policy = policy.process_state(state, epsilon=epsilon, distribution=True, action=True)
        distances.append(kl_divergence(q_trajectory, q_policy))
    return distances


def values_to_distribution(values):
    policy = {}
    for state in values:
        policy[state] = softmax(values[state])
    return policy


def kl_divergence_norm(traj, pol, actions, epsilon=0.):
    p = pol.q_table
    # kl divergence using epsilon-greedy policies
    # aggregates all divergences by averaging them
    p_traj = traj_to_policy(traj, actions)
    # p1 = values_to_policy(p1, epsilon)
    policy = values_to_policy(p, epsilon)

    distances = []
    for i, state in enumerate(p_traj):
        if state not in policy:
            add_dummy_q(policy, state, actions, epsilon)
        qp1 = p_traj[state]
        qp2 = policy[state]
        # print(f'Best action for traj and policy, state {i}: {np.argmax(qp1)} - {np.argmax(qp2)}')
        distances.append(kl_divergence(qp1, qp2))
    return mean(distances)


def kl_divergence_norm_softmax(traj, pol, actions):
    # copy paste of kl divergence but with softmax
    # because I'm lazy
    p = pol.q_table
    p_traj = traj_to_policy(traj, actions)
    # p1 = values_to_policy(p1, epsilon)
    policy = values_to_distribution(p)
    # print(p)
    distances = []
    for i, state in enumerate(p_traj):
        # print(state)
        if state not in policy:
            add_dummy_policy(policy, state, actions)
        qp1 = p_traj[state]
        qp2 = policy[state]
        # print(f'Best action for traj and policy, state {i}: {np.argmax(qp1)} - {np.argmax(qp2)}')
        distances.append(kl_divergence(qp1, qp2))
    return mean(distances)


def add_dummy_policy(policy, state, actions):
    # returns a dummy behavior in case a state has not been visited
    # when running a tabular policy
    n_actions = len(actions)
    policy[state] = [1./n_actions for _ in range(n_actions)]


def add_dummy_q(policy, state, actions, epsilon=0.):
    # same as add_dummy_policy, but for q-values
    n_actions = len(actions)
    policy[state] = [1e-6 + epsilon/n_actions for _ in range(n_actions)]
    best_random_action = random.choice(range(n_actions))
    policy[state][best_random_action] = 1. - epsilon - 1e-6*(n_actions-1)


def traj_to_policy(traj, actions, epsilon=0.):
    # converts a trajectory from a planner to a policy
    # where the taken action has 99.99999% probability
    trajectory_as_policy = {}
    for state, action in traj:
        action_index = actions.index(action)
        actions_len = len(actions)
        qs = [1e-6 + epsilon/actions_len for _ in range(actions_len)]
        qs[action_index] = 1. - 1e-6 * (actions_len-1) - epsilon
        trajectory_as_policy[tuple(state)] = qs
    return trajectory_as_policy


def values_to_policy(policy, epsilon=0.):
    policy_table = {}
    for s in policy.keys():
        q = policy[s]
        q_length = len(q)
        policy_table[s] = [1e-6 + epsilon/q_length for _ in range(q_length)]
        policy_table[s][np.argmax(q)] = 1. - 1e-6*(q_length-1) - epsilon
    return policy_table


def plot_mean_divergence(goal, eps, *divs):
    # given a list of divergences, plot all as a bar plot.
    fig, ax = plt.subplots()
    plt.title(f'Divergence between trajectory for goal {goal} and policies. Eps = {eps}')
    goals_text = [f'p{n}' for n in range(len(divs))]
    plt.bar(goals_text, divs)
    save_path = os.path.abspath('../imgs')
    for i, d in enumerate(divs):
        ax.text(i, d + 1.5, f'{d:.2f}')
    ax.set_ylim([0., 100.])
    plt.savefig(f'{save_path}/goal_{goal}_eps_{eps}.jpg')

# def plot_traj_policy_divergence(goal, eps, *kls):
#     # not used for now
#     fig = plt.figure()
#     ax = fig.add_axes([0,0,1,1])
#     combinations = [f'KL(t{goal}, p{n}' for n in range(len(kls))]
#     ax.bar(combinations, kls)
#     ax.set(title=f'Eps = {eps}')
#     plt.show()
