from typing import Any, Dict, List, Sequence, Collection, Tuple
from pddlgym.core import PDDLEnv
from pddlgym.structs import Predicate, Literal
from pddlgym.parser import PDDLProblemParser
from gym import Env
from env.indexing import compute_indices
import numpy as np
import itertools
from collections import defaultdict

# Wrapper class for a PDDLProblem


class PDDLProblem():

    def __init__(self, wrapped: PDDLProblemParser) -> None:
        self.wrapped = wrapped
        self._all_ground_literals = set()
        # Initialize all ground literals
        self._type_to_objs = defaultdict(list)

        for obj in sorted(self.wrapped.objects):
            self._type_to_objs[obj.var_type].append(obj)

        for predicate in self.wrapped.predicates.values():
            choices = [self._type_to_objs[vt] for vt in predicate.var_types]
            for choice in itertools.product(*choices):
                if len(set(choice)) != len(choice):
                    continue
                lit = predicate(*choice)
                self._all_ground_literals.add(lit)

    @property
    def objects(self):
        return self.wrapped.objects

    @property
    def objectmap(self):
        return self.wrapped

    @property
    def predicates(self) -> Predicate:
        return self.wrapped.predicates

    @property
    def all_ground_literals(self) -> Collection[Literal]:
        return self._all_ground_literals

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
        self.env = wrapped_env
        self.problems = [PDDLProblem(problem) for problem in wrapped_env.problems]

    def seed(self, seed: int) -> List[int]:
        return self.env.seed(seed=seed)

    def reset(self) -> Any:
        return self.env.reset()

    @property
    def observation_space(self) -> np.ndarray:
        return to_flat_dense_binary(self.problems[self.env._problem_idx].all_ground_literals, self.env.problems[self.env._problem_idx])
        # return to_flat_dense_binary(self.env.observation_space.predicates, self.env.problems[self.env._problem_idx])

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        state, reward, done, debug_info = self.env.step(action)
        array_state = to_flat_dense_binary(state, self.env.problems[self.env._problem_idx])
        return array_state, reward, done, debug_info

    def close(self) -> None:
        return self.env.close()

    def render(self, *args, **kwargs) -> None:
        return self.env.render(args, kwargs)