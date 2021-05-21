import pddlgym
from pddlgym_planners.ff import FF

# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvBlocks-v0")
env.fix_problem_index(5)
obs, debug_info = env.reset()
print(env.observation_space.__dict__)
print(obs.goal)
planner = FF()
plan = planner(env.domain, obs)
print(plan)
env.step(plan[1])
print(env.observation_space.__dict__)
print(len(env.action_space.all_ground_literals(obs, valid_only=False)))
# for act in plan:
#     obs, reward, done, info = env.step(act)
#     print(info)
