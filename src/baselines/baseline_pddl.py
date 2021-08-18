#!python
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join('.')))
# from pddlgym_planners.fd import FD
# from pddlgym.core import InvalidAction
from typing import Callable
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


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == "__main__":
    pddl_env = pddlgym.make("PDDLEnvBlocks-v0")
    # pddl_env = pddlgym.make("PDDLEnvSokoban-v0")
    pddl_env.fix_problem_index(0)
    env = PDDLGymVecWrapper(pddl_env, only_valid_actions=False)
    # print(env.print_debug())
    # env = gym.make("CartPole-v1") # To check what kind of object we get as an observation here

    # model = PPO("MlpPolicy", env, verbose=1)
    # model = DQN("MlpPolicy", env, verbose=1)
    # model = DQN("MlpPolicy", env, verbose=1, learning_rate=linear_schedule(0.2), learning_starts=0, exploration_initial_eps=1, exploration_final_eps=0.01)
    model = A2C("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=500000)
    print("Goal Reached: ", env.goal_counter)

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
