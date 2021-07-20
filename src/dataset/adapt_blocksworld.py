import sys
import os

BLOCKSWORLD_MAPPING = {
    "pick-up": "pickup",
    "handempty": "handempty robot",
}

ACTIONS = ["pickup", "putdown", "unstack", "stack"]
OBJECTS = []

def gen_actions(objects):
    putdown = []
    unstack = []
    pickup = []
    stack = []
    blocks = objects['block']
    for i in range(len(blocks)):
        putdown.append(f'(putdown {blocks[i]})')
        unstack.append(f'(unstack {blocks[i]})')
        pickup.append(f'(pickup {blocks[i]})')
        for j in range(len(blocks)):
            if i == j:
                continue
            stack.append(f'(stack {blocks[i]} {blocks[j]})')
    return '\n'.join(putdown), '\n'.join(unstack), '\n'.join(pickup), '\n'.join(stack)


def create_problem_files(parser, template_name, goals, path):
    domain_name = path.split('/')[-1]
    files = []
    parser.parse_problem(template_name)
    with open(goals, 'r') as g:
        goals_text = list(map(lambda x: x.rstrip('\n'), g.readlines()))
        for i, goal in enumerate(goals_text):
            files.append(write_problem_file(parser, goal, i, domain_name, path))
    return files

def build_objects(parser):
    objects_str = ''
    for t in parser.objects:
        objs = ' '.join(parser.objects[t])
        objects_str += f'{objs} - {t}\n'
    objects_str += 'robot - robot\n'
    return objects_str

def build_initial_state(parser):
    putdown, unstack, pickup, stack =  gen_actions(parser.objects)
    literals = ''
    for t in parser.state:
        predicate = t[0]
        # objs = t[1:]
        if predicate == 'handempty':
            literals += '(handempty robot)\n'
        else:
            literal = ' '.join(t)
            literals += f'({literal})\n'
    return f"{putdown}\n\
        {unstack}\n\
        {pickup}\n\
        {stack}\n\
        {literals}"

    
def write_problem_file(parser, goal, i, domain, path):
    file_path = f'{path}/{domain}_problem_{i:02d}.pddl'
    goal = ' '.join(goal.split(','))
    problem_definition = f"(define (problem blocks_words)\n\
        (:domain {parser.domain_name})\n\
        (:objects {build_objects(parser)})\n\
        (:init {build_initial_state(parser)})\n\
        (:goal (and {goal})))"
    with open(file_path, 'w') as f:
        
        f.write(problem_definition)
    
    return file_path

def adapt_observation(obs_path):
    with open(obs_path, 'r') as f:
        actions = f.readlines()
    with open(obs_path, 'w') as f:
        for action in actions:
            action_split = action.rstrip(')\n').split(' ')
            a = action_split[0].lower().lstrip('(')
            objs = action_split[1:]
            if a == 'pick-up':
                a = 'pickup'
                objs = objs[0]
            elif a == 'put-down':
                a = 'putdown'
                objs = objs[0]
            elif a == 'unstack':
                objs = objs[0]
            elif a == 'stack':
                objs = ' '.join(objs)
            f.write(f'({a} {objs})\n')