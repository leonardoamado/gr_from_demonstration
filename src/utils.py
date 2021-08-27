from typing import Collection
from pddlgym.core import InvalidAction, PDDLEnv
from pddlgym.structs import Literal

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True


def find_action(action_name: str, actions: Collection[Literal]) -> Literal:
    """ Finds an action in a set collection of actions generared from PDDLEnv
    """
    for action in actions:
        if action.__repr__() == action_name:
            return action
    return None


def find_actions(action_names: Collection[str], actions: Collection[Literal]) -> Collection[Literal]:
    literals = []
    for action in actions:
        if action.__repr__() in action_names:
            literals.append(action)
    return literals


def solve_fset(fset):
    '''
    Converts a fronzenset to an ordered tuple.
    '''
    return tuple(sorted(tuple(fset)))


if __name__ == "__main__":
    env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/',raise_error_on_invalid_action=RAISE_ERROR_ON_VALID, dynamic_action_space=DYNAMIC_ACTION_SPACE)
    env.fix_problem_index(0)
    init, _ = env.reset()
    print(solve_fset(init.literals))
