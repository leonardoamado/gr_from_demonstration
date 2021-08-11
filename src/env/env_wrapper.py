from typing import Any, Dict, List, Sequence, Collection, Set, Tuple
from gym.spaces.space import Space
from gym.spaces.discrete import Discrete
from pddlgym.core import PDDLEnv
from pddlgym.spaces import LiteralActionSpace, LiteralSetSpace, LiteralSpace
from pddlgym.structs import Predicate, Literal, State
from pddlgym.parser import PDDLProblemParser
from gym import Env
from env.indexing import compute_indices
import numpy as np
import itertools
from collections import defaultdict

# Wrapper class for a PDDLProblem


class PDDLProblem():

    def __init__(self, problem: PDDLProblemParser) -> None:
        self._problem = problem
        self._all_ground_literals = set()
        # Initialize all ground literals
        self._type_to_objs = defaultdict(list)

        for obj in sorted(self._problem.objects):
            self._type_to_objs[obj.var_type].append(obj)

        for predicate in self._problem.predicates.values():
            choices = [self._type_to_objs[vt] for vt in predicate.var_types]
            for choice in itertools.product(*choices):
                if len(set(choice)) != len(choice):
                    continue
                lit = predicate(*choice)
                self._all_ground_literals.add(lit)

    @property
    def objects(self):
        return self._problem.objects

    @property
    def objectmap(self):
        return self._problem

    @property
    def predicates(self) -> Predicate:
        return self._problem.predicates

    @property
    def all_ground_literals(self) -> Collection[Literal]:
        return self._all_ground_literals

    @property
    def initial_state(self) -> Set:
        return self._problem.initial_state
    
    @property
    def goal(self) -> Set:
        return self._problem.goal


class LiteralSpaceWrapper(Discrete):
    r""" A wrapper for LiteralActionSpace from PDDLGym to work with the baselines"""
    def __init__(self, wrapped_space: LiteralSpace, env: Env) -> None:
        # assert isinstance(wrapped_env, LiteralActionSpace)
        self._space = wrapped_space
        self_env = env
        super().__init__(len(self._space.all_ground_literals(State(env.get_problem().initial_state, env.get_problem().objects, env.get_problem().goal))))  # TODO Don't do this only with the initial state

    def sample(self) -> int:  # TODO Check how the sampling in PDDLGym works
        return self._space.sample()

    def contains(self, x: Any) -> bool:
        return self._space.contains(x)

    def __repr__(self) -> str:
        return self._space.__repr__()

    def __eq__(self, o: object) -> bool:
        return self._space.__eq__(o)
# What follows was adapted from pddl_env


def to_dense_binary(literals: Sequence[Collection[Predicate]],
                    problem: PDDLProblem,
                    dtype: type = np.float32) -> Dict[int, np.ndarray]:
    # indices, shapes = compute_indices(literals, problem.objectmap.objects, problem.predicates)
    indices, shapes = compute_indices(literals, problem.objects, problem.predicates.values())

    n = len(literals)
    features = {arity: np.zeros((n,) + shape, dtype=dtype) for arity, shape in shapes.items()}
    for k, idx in indices.items():
        features[k][idx] = 1

    return features


def to_flat_dense_binary(literals: Sequence[Collection[Literal]],
                         problem: PDDLProblem,
                         dtype: type = np.float32) -> np.ndarray:
    features = to_dense_binary(literals, problem, dtype=dtype)
    return np.concatenate(tuple(x.reshape((x.shape[0], -1)) for x in features.values()), axis=-1)

# Below is the new stuff


class PDDLGymVecWrapper(Env):
    """
    A wrapper class for pddlgym.core.PDDLEnv that can return states and actions as arrays suitable for
    [RL Baselines](https://github.com/DLR-RM/stable-baselines3).

    Vectorization logic adapted from the [pddlenv](https://github.com/gehring/pddlenv) project following [discussions on the PDDLGym project](https://github.com/tomsilver/pddlgym/issues/58)
    """

    def __init__(self, wrapped_env: PDDLEnv) -> None:
        super().__init__()
        self._env = wrapped_env
        self._problems = [PDDLProblem(problem) for problem in wrapped_env.problems]
        self._all_ground_literals = self._problems[self._env._problem_idx].all_ground_literals
        # self._all_ground_action_literals = self._problems  # TODO Check that I instantiate all possible actions at this point
        # self._observation_space = to_flat_dense_binary(self._all_ground_literals, self._env.problems[self._env._problem_idx])
        self._observation_space = LiteralSpaceWrapper(wrapped_env.observation_space, self)
        self._action_space = LiteralSpaceWrapper(wrapped_env.action_space, self)

    def seed(self, seed: int) -> List[int]:
        return self._env.seed(seed=seed)

    def reset(self) -> np.ndarray:
        state, _debug = self._env.reset()
        vec_state = to_flat_dense_binary(state.literals, self.get_problem())
        return vec_state

    def get_state(self) -> State:
        return self._env.get_state()

    def get_problem(self) -> PDDLProblem:
        return self._problems[self._env._problem_idx]

    @property
    def observation_space(self) -> np.ndarray:
        return self._observation_space
        # return to_flat_dense_binary(self.env.observation_space.predicates, self.env.problems[self.env._problem_idx])

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, debug_info = self._env.step(action)
        # array_state = to_flat_dense_binary(state, self.env.problems[self.env._problem_idx])
        array_state = to_flat_dense_binary(state.literals, self._problems[self._env._problem_idx])
        return array_state, reward, done, debug_info

    def close(self) -> None:
        return self._env.close()

    def render(self, *args, **kwargs) -> None:
        return self._env.render(args, kwargs)