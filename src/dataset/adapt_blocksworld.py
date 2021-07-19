BLOCKSWORLD_MAPPING = {
    "pick-up": "pickup",
    "handempty": "handempty robot",
}

ACTIONS_MAPPING = {

}

ACTIONS = ["pickup", "putdown", "unstack", "stack"]
OBJECTS = []

def gen_actions(blocks):
    putdown = []
    unstack = []
    pickup = []
    stack = []
    for i in range(len(blocks)):
        putdown.append(f'(putdown {blocks[i]})')
        unstack.append(f'(unstack {blocks[i]})')
        pickup.append(f'(pickup {blocks[i]})')
        for j in range(len(blocks)):
            if i == j:
                continue
            stack.append(f'(stack {blocks[i]} {blocks[j]})')
    return '\n'.join(putdown), '\n'.join(unstack), '\n'.join(pickup), '\n'.join(stack)

def create_problem_files(template, goals, path):
    domain_name = path.split('/')[-1]
    files = []
    with open(template, 'r') as t, open(goals, 'r') as g:
        template_text = list(filter(lambda x: x != '', map(lambda x: x.rstrip('\n'), t.readlines())))
        goals_text = list(map(lambda x: x.rstrip('\n'), g.readlines()))
        for i, goal in enumerate(goals_text):
            files.append(write_problem_file(goal, i, domain_name, template_text, path))
    return files

def extract_objects(objects):
    return objects.split(' - block')[0].split(' ')

def write_problem_file(goal, i, domain, template, path):
    objects = []
    file_path = f'{path}/{domain}_problem_{i:02d}.pddl'
    with open(file_path, 'w') as f:
        for line in template:

            if ' - block' in line:
                objects = extract_objects(line)
                if ')' in line:
                    line = line.rstrip(')') + ' robot - robot)'
                else:
                    line += ' robot - robot'
            elif '(:init' in line:
                putdown, unstack, pickup, stack =  gen_actions(objects)
                line = '\n'.join([line, '\n', putdown, unstack, pickup, stack])
            elif 'HANDEMPTY' in line:
                line = '(handempty robot)'
            elif '<HYPOTHESIS>' in line:
                line = ' '.join(goal.split(','))
            f.write(f'{line}\n')
    return file_path

def adapt_obs(obs_path):
    with open(obs_path, 'r') as f:
        actions = f.readlines()
    with open(obs_path, 'w') as f:
        for action in actions:
            action_split = action.rstrip(')\n').split(' ')
            a = action_split[0].lower().lstrip('(')
            objs = action_split[1:]
            if a == 'pick-up':
                print(a, objs)
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