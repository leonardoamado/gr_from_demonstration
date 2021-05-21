from matplotlib import pyplot as plt
import numpy as np
from math import log2

from numpy.core.fromnumeric import mean

def kl_divergence(p1, p2):
    return sum(p1[i] * log2(p1[i]/p2[i]) for i in range(len(p1)))

def kl_divergence_per_plan_state(trajectory, p1, p2):
    p1 = values_to_policy(p1)
    p2 = values_to_policy(p2)
    per_state_divergence = []
    for state in trajectory:
        if state not in p1:
            p1[state] = [0.000001 for _ in range(24)]
        if state not in p2:
            p2[state] = [0.000001 for _ in range(24)]
        qp1 = p1[state]
        qp2 = p2[state]
        per_state_divergence.append(kl_divergence(qp1, qp2))
    return per_state_divergence


def kl_divergence_norm(p1, p2):
    distances = []
    for state in p1:
        if state not in p2:
            continue
        qp1 = p1[state]
        qp2 = p2[state]
        distances.append(kl_divergence(qp1, qp2))
    return mean(distances)


def kl_divergence_mean(p1, p2):

    pass

def values_to_policy(policy):
    policy_table = {}
    for s in policy.keys():
        q = policy[s]
        policy_table[s] = [0.0000001 for _ in range(len(q))]
        policy_table[s][np.argmax(q)] = 1. - 2.3e-05
    return policy_table


def plot(goal, *kls):
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    combinations = [f'KL(t{goal}, p{n}' for n in range(3)]
    ax.bar(combinations, kls)
    plt.show()