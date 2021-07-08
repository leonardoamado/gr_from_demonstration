import pddlgym
from pddlgym_planners.fd import FD

# See `pddl/sokoban.pddl` and `pddl/sokoban/problem3.pddl`.
env = pddlgym.make("PDDLEnvBlocks-v0", dynamic_action_space=True)
env.fix_problem_index(0)
obs, debug_info = env.reset()
print(debug_info)
# print(env.observation_space.__dict__)
# print(obs.goal)
# print(obs.literals == obs.literals)
planner = FD()
plan = planner(env.domain, obs)
# print(plan)
env.step(plan[1])
# print(env.__dict__)
print(env.domain.predicates)
# print(env.observation_space.__dict__)
# print(len(env.action_space.all_ground_literals(obs, valid_only=False)))
# for act in plan:
#     obs, reward, done, info = env.step(act)
#     print(info)