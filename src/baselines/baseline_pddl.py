#!python
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join('.')))
# from pddlgym_planners.fd import FD
# from pddlgym.core import InvalidAction
import gym
import pddlgym
from env.env_wrapper import PDDLGymVecWrapper
from stable_baselines3 import PPO, DQN, A2C
import matplotlib
matplotlib.use('TkAgg') # We need this for the environment to render in Mac
# matplotlib.use('nbAgg')
# matplotlib.use('WebAgg')


# configs for environment

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True


if __name__ == "__main__":

    pddl_env = pddlgym.make("PDDLEnvBlocks-v0")
    # pddl_env = pddlgym.make("PDDLEnvSokoban-v0")
    pddl_env.fix_problem_index(0)
    env = PDDLGymVecWrapper(pddl_env)
    # env = gym.make("CartPole-v1") # To check what kind of object we get as an observation here

    # model = PPO("MlpPolicy", env, verbose=1)
    # model = DQN("MlpPolicy", env, verbose=1)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=50000)

    obs = env.reset()
    print("S*", env._initialStates[0].goal)
    print("S0", env._observation_space.vec_to_literals(obs))
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        print("A%d: %s" % (i, env.action_space.i_to_literal(action)))
        print("S%d: %s" % (i+1, env._observation_space.vec_to_literals(obs)))
        env.render()
        if done:
            obs = env.reset()
            break

    env.close()
