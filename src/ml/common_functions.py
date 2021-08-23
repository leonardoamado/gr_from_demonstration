import numpy as np
import pddlgym
from ml.common import PARTIAL_GOAL_REWARD


def extract_offsets(blocks, predicates):
    print('blocks order:', blocks)
    print('predicate order:')
    offsets = {}
    i = 0
    for pred in predicates:
        print('\t', pred)
        offsets[pred.name] = i
        num_lits = 1
        for t in pred.var_types:
            if t == 'block':
                num_lits *= len(blocks)
        i += num_lits
    state_size = i
    return state_size, offsets


def build_state(obs, state_size, offsets, blocks):
        state = [0. for _ in range(state_size)]
        ground_literals = obs[0]
        for lit in ground_literals:
            base_offset = offsets[lit.predicate.name]
            var_offset = 1
            vars = lit.variables
            if len(vars) == 2:
                idx_first = blocks.index(vars[0].name)
                idx_second = blocks.index(vars[1].name)
                var_offset = len(blocks) * idx_first + idx_second
            else:
                v = vars[0]
                if v.var_type == 'block':
                    var_offset = blocks.index(v.name)
                else:
                    var_offset = 0
            # for v in lit.variables:
            #     if v.var_type == 'block':
            #         if len(lit.variables) == 2:
            #             var_offset = 
            #         # var_offset *= self.blocks.index(v.name)
            #     else:
            #         var_offset = 0
            # print('Offsets:',base_offset, var_offset, base_offset + var_offset)
            state[base_offset + var_offset] = 1.
        return np.array([state])

def build_state_linear(obs, state_size, offsets, blocks, action_space, action=None):
    state = [0. for _ in range(state_size)]
    ground_literals = obs[0]
    for lit in ground_literals:
        base_offset = offsets[lit.predicate.name]
        var_offset = 1
        vars = lit.variables
        if len(vars) == 2:
            idx_first = blocks.index(vars[0].name)
            idx_second = blocks.index(vars[1].name)
            var_offset = len(blocks) * idx_first + idx_second
        else:
            v = vars[0]
            if v.var_type == 'block':
                var_offset = blocks.index(v.name)
            else:
                var_offset = 0
        # for v in lit.variables:
        #     if v.var_type == 'block':
        #         if len(lit.variables) == 2:
        #             var_offset = 
        #         # var_offset *= self.blocks.index(v.name)
        #     else:
        #         var_offset = 0
        # print('Offsets:',base_offset, var_offset, base_offset + var_offset)
        state[base_offset + var_offset] = 1.
    state.extend([0. for _ in range(action_space)])
    if action:
        state[state_size + action] = 1.
    # print(state)
    return np.array([state])


def extract_objects(obs):
    objects = {}
    obs_objects = obs[1]
    for object in obs_objects:
        if object.var_type not in objects:
            objects[object.var_type] = []
        objects[object.var_type].append(object.name)
    return objects


def check_for_partial_goals(obs, goal_literals_achieved):
        literals = obs[0]
        goals = obs[2].literals
        r = 0.
        for lit in literals:
            if lit not in goal_literals_achieved and lit in goals:
                r += PARTIAL_GOAL_REWARD
                goal_literals_achieved.add(lit)
        return r


def filter_valid_actions(obs):
    pddlgym