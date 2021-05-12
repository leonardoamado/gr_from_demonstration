import pddlgym
from pddlgym_planners.ff import FF

# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvSokoban-v0")
env.fix_problem_index(2)
obs, debug_info = env.reset()
print(env.observation_space.__dict__)
planner = FF()
plan = planner(env.domain, obs)
for act in plan:
    obs, reward, done, info = env.step(act)
