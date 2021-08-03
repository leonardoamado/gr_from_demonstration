#!python
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join('.')))
# from pddlgym_planners.fd import FD
# from pddlgym.core import InvalidAction
from stable_baselines3 import PPO, DQN
from env.env_wrapper import PDDLGymVecWrapper
import gym
import pddlgym

# configs for environment

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True


# env = gym.make("CartPole-v1")
# env = pddlgym.make("PDDLEnvBlocks-v0",
#                    raise_error_on_invalid_action=RAISE_ERROR_ON_VALID,
#                    dynamic_action_space=DYNAMIC_ACTION_SPACE
#                    )

# env = pddlgym.make("PDDLEnvBlocks-v0")
pddl_env = pddlgym.make("PDDLEnvSokoban-v0")
pddl_env.fix_problem_index(2)
env = PDDLGymVecWrapper(pddl_env)

# model = PPO("MlpPolicy", env, verbose=1)
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()

env.close()
