from re import template
import sys
import os
# sys.path.append(os.path.abspath(os.path.join('..')))
# sys.path.append(os.path.abspath(os.path.join('.')))
import pyperplan.planner as pyperplan
import pyperplan.pddl.parser as pyparser
import pyperplan.grounding as grounding
from pathlib import Path
import time
import random
import pickle

from pddlgym_planners.fd import FD

from pddlgym.core import InvalidAction, PDDLEnv
from utils import solve_fset

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True


# very silly implementation to get action names, replace this after a good parser is found.
def parse_action(domain):
    domain.seek(0)
    action_list = []
    for line in domain:
        if '(:action' in line:
            action_list.append(line.replace(' ', '').replace('\n', '').split('action')[1])
    return action_list


# Adds to a PDDL domain the necessary line to run on PDDLGym
def complete_domain(domain, path, parser):
    action_list = parse_action(domain)
    domain.seek(0)
    print(path +"/domain.txt")
    new_domain = open(path +"/domain.pddl", "w")
    action_string = '; (:actions'
    for action in action_list:
        action_string += ' '
        action_string += action
    action_string += ')'
    counter = 0
    pred_counting = False
    for line in domain:
        new_domain.write(line)
        if '(:predicates' in line:
            pred_counting = True
        if pred_counting:
            if '(': counter += line.count('(') 
            if ')': counter -= line.count(')') 
            #print(line + '' + str(counter))
        if pred_counting and counter == 0:
            new_domain.write('')
            new_domain.write(action_string)
            new_domain.write('')
            pred_counting = False
    new_domain.close()


# Converts a pyperplan predicate to a PDDL string
def pred_to_string(predicate):
    st = '(' + predicate.name
    for sig in predicate.signature:
        st += ' '
        st += sig[0]
    st += ')'
    return st


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


def complete_problem(problem_file, problem, task, goal, number, path):
    problem_file.seek(0)
    print('Writing on: ' + path +"/problems/problem" + str(number) + ".pddl")
    new_problem = open(path + "/problems/problem" + str(number) + ".pddl", "w")
    action_string = '; action literals'
    counter = 1
    init_state = False
    for line in problem_file:
        if init_state and counter >0:
            if '(': counter += line.count('(') 
            if ')': counter -= line.count(')') 
        if init_state and counter >0:            
            continue
        init_state = False
        if '<HYPOTHESIS>' in line:
            new_problem.write('\t' + goal)
            continue
        new_problem.write(line)
        if '(:init' in line:
            init_state = True
            
            for pred in problem.initial_state:
                new_problem.write('\t' + pred_to_string(pred))
                new_problem.write('\n')
            new_problem.write('\n')
            new_problem.write(action_string)
            new_problem.write('\n\t')
            for t in task:
                new_problem.write(t.name)
                new_problem.write('\n\t')
    new_problem.close()


def complete_obs(observations, path, name):
    new_obs = open(path + name, "w")
    for line in observations:
        new_obs.write(line)
    new_obs.close()
        

"""
This is the main method that converts a GR problem directory to a PDDLGym 
version of the PDDL

@note Assumption: We assume that the format used in the directory is the same
that Ramirez and Geffener use

@param d            The directory of the problem
@param output       The output directory
@param obs_per      The percentile of available observation
"""
def gr_to_gym(d, output='output', obs_per=100):
    domain = open(d + "/domain.pddl", "r")
    hypothesis = open(d+ "/hyps.dat")
    correct_goal = open(d+ "/real_hyp.dat")
    observations = open (d + "/obs.dat")
    template = open(d + "/template.pddl")

    #parse the domain file, generating a task (grounded actions)
    problem = pyperplan._parse(d + "/domain.pddl", d + "/template.pddl")
    task = grounding.ground_no_goal(problem)
    
    #Setup directories, still needs some fixing here.
    Path(output +'/'+ d).mkdir(parents=True, exist_ok=True)
    Path(output +'/'+ d + '/problems').mkdir(parents=True, exist_ok=True)

    #Complete domain and multiple problems
    goals = []
    for line in hypothesis:
        goals.append(line)
    complete_domain(domain, output +'/' + d, None)
    for count, goal in enumerate(goals):
        complete_problem(template, problem, task, goal, count, output +'/' + d)
    
    #Copy the observations and the correct goal to the desired location
    complete_obs(observations, output +'/' + d + '/', 'obs.dat')
    complete_obs(correct_goal, output +'/' + d + '/', 'real_hyp.dat')    


def gr_to_gym_custom_obs(d, output='output'):
    domain = open(d + "/domain.pddl", "r")
    hypothesis = open(d+ "/hyps.dat")
    correct_goal = open(d+ "/real_hyp.dat")
    observations = open (d + "/obs.dat")
    template = open(d + "/template.pddl")

    #parse the domain file, generating a task (grounded actions)
    problem = pyperplan._parse(d + "/domain.pddl", d + "/template.pddl")
    task = grounding.ground_no_goal(problem)
    
    #Setup directories, still needs some fixing here.
    Path(output +'/'+ d).mkdir(parents=True, exist_ok=True)
    Path(output +'/'+ d + '/problems').mkdir(parents=True, exist_ok=True)

    #Complete domain and multiple problems
    goals = []
    for line in hypothesis:
        goals.append(line)
    complete_domain(domain, output +'/' + d, None)
    for count, goal in enumerate(goals):
        complete_problem(template, problem, task, goal, count, output +'/' + d)
    #Copy the observations and the correct goal to the desired location
    complete_obs(observations, output +'/' + d + '/', 'obs.dat')
    complete_obs(correct_goal, output +'/' + d + '/', 'real_hyp.dat')


def create_observabilities(d, output, ind=0):
    print(output +'/'+ d + '/problems')
    print(d + "/domain.pddl")
    env = PDDLEnv(d + "/domain.pddl", d + '/problems',raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                            dynamic_action_space=DYNAMIC_ACTION_SPACE)
    planner = FD()
    traj = []
    env.fix_problem_index(ind)
    init, _ = env.reset()
    #print(f'GOAL {init.goal}')
    
    # traj is an action pair tuple, need to map this to state action number pair
    plan = planner(env.domain, init)
    traj = []
    obs_list = [0.1,0.3,0.5,0.7,1.0]
    traj_list = {}
    for a in plan:
        state_action_pair = (solve_fset(init.literals), a)
        traj.append(state_action_pair)
        init, _, _, _ = env.step(a)
    for obs in obs_list:
        traj_list[obs] = remove_obs(traj, obs)
        save_obs(traj_list[obs], output +'/' +'obs' + str(obs)+'.dat')
        with open(output +'/' +'obs' + str(obs)+'.pkl', "wb") as output_file:
            pickle.dump(traj_list[obs], output_file)


def save_obs(traj, out):
    new_obs = open(out, "w")
    for line in traj:
        str_out = '' 
        for pred in line[0]:
            str_out += str(pred) + ' '
        str_out += ';'
        str_out += str(line[1])
        new_obs.write(str_out)
        new_obs.write('\n')
    new_obs.close()


if __name__ == "__main__":
    create_observabilities('output/blocks_gr', 'output/blocks_gr')
    #gr_to_gym_new('dummy_gr', 'output', 100)
    #TODO create a complete main here