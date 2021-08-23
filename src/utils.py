from pddlgym.core import InvalidAction, PDDLEnv

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True

'''
Converts a fronzenset to an ordered tuple.
'''
def solve_fset(fset):
    return tuple(sorted(tuple(fset)))

if __name__ == "__main__":
    env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/',raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
                            dynamic_action_space=DYNAMIC_ACTION_SPACE)
    env.fix_problem_index(0)
    init, _ = env.reset()
    print(solve_fset(init.literals))