BLOCKSWORLD_MAPPING = {
    "pick-up": "pickup",
    "handempty": "handempty robot",
}

ACTIONS_MAPPING = {

}

ACTIONS = ["pickup", "putdown", "unstack", "stack"]
OBJECTS = []


def replace_words(file):
    with open(file, 'w+') as f:
        new_lines = []
        for line in f.readlines():
            if " - block" in line:
                objects = line.split(' - block')[0].split(' ')

            lowercase_line = line.lower()
            for key, val in BLOCKSWORLD_MAPPING.items():
                lowercase_line = lowercase_line.replace(key, val)
            new_lines.append(lowercase_line)



# def add_actions_to_problem(file):
#     for

def gen_actions(blocks):
    putdown = []
    unstack = []
    pickup = []
    stack = []
    for i in range(len(blocks)):
        putdown.append(f'(putdown {blocks[i]})')
        unstack.append(f'(unstack {blocks[i]})')
        for j in range(len(blocks)):
            if i == j:
                continue
            pickup.append(f'(pickup {blocks[i]} {blocks[j]})')
            stack.append(f'(stack {blocks[i]} {blocks[j]})')
    return '\n'.join(putdown), '\n'.join(unstack), '\n'.join(pickup), '\n'.join(stack)

def add_goal(goals, f):
    for goal in goals.readlines():
        pass

def create_problem_files(template, goals):
    with open(template, 'r') as t, open(goals, 'r') as g:
        template_text = list(filter(lambda x: x != '', map(lambda x: x.rstrip('\n'), t.readlines())))
        goals_text = list(map(lambda x: x.rstrip('\n'), g.readlines()))
        for i, goal in enumerate(goals_text):
            write_problem_file(goal, i, template_text)
        for i in range(len(template_text)):
            pass

def write_problem_file(goal, i, template, domain, problem):
    objects = []
    with open(f'{domain}_{problem}_{i:02d}', 'w') as f:
        # template = f.readlines()
        for line in template:

            if ' - block' in line:
                objects = extract_objects(line)
            elif '(:init' in line:
                putdown, unstack, pickup, stack =  gen_actions(objects)
                line = '\n'.join([line, '\n', putdown, unstack, pickup, stack])
            elif '<HYPOTHESIS>' in line:
                line = goal
            f.write(line)

            pass
    pass

def extract_objects(objects):
    return objects.split(' - block')[0].split(' ')

