import sys
import pddlgym
print(sys.path)
sys.path.append('..')
from pddlgym_planners.fd import FD


class EnvManager:
    """
    Class used to manage PDDLGym environments.
    Not really used throughout the code,
    since it is easier to just interact directly
    with the environment object.
    """
    def __init__(self, envname):
        self.env = pddlgym.make(f'PDDLEnv{envname}-v0')
        self.initial_obs = None
        self.current_obs = None
        self.planner = None

    def reset(self, problem=None):
        if problem:
            self.env.fix_problem_index(problem)
        obs, info = self.env.reset()
        self.initial_obs = obs
        self.current_obs = obs
        self.current_info = info
        return obs, info

    def problems(self):
        return self.env.problems

    def plan(self):
        if self.planner is None:
            self.planner = FD()
        plan = self.planner(self.env.domain, self.initial_obs)
        return plan

    def step(self, a):
        obs, r, done, info = self.env.step(a)
        self.current_obs = obs

    def execute_plan(self, plan):
        obss = []
        for a in plan:
            obs, r, done, info = self.env.step(a)
            obss.append(obs)
        return obss

    def render(self):
        return self.env.render()
