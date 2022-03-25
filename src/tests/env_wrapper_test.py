#!python

import gym
import pddlgym
from env.env_wrapper import PDDLGymVecWrapper
from stable_baselines3 import PPO, DQN

import unittest

# configs for environment

RAISE_ERROR_ON_VALID = False
DYNAMIC_ACTION_SPACE = True


class TestListElements(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("*********Testing List Elements*********")

    def setUp(self):
        print("Unit tests for all_literals")

    def test_PDDLEnvSokoban_v0(self):
        pddl_env = pddlgym.make("PDDLEnvSokoban-v0")
        pddl_env.fix_problem_index(2)
        env1 = PDDLGymVecWrapper(pddl_env)
        all_literals1 = env1.all_ground_literals
        self.all_literals1 = all_literals1

        pddl_env = pddlgym.make("PDDLEnvSokoban-v0")
        pddl_env.fix_problem_index(2)
        env2 = PDDLGymVecWrapper(pddl_env)
        all_literals2 = env2.all_ground_literals
        self.all_literals2 = all_literals2
        self.assertCountEqual(self.all_literals1, self.all_literals2)

    def test_PDDLEnvBlocks_v0(self):
        pddl_env = pddlgym.make("PDDLEnvBlocks-v0")
        pddl_env.fix_problem_index(2)
        env1 = PDDLGymVecWrapper(pddl_env)
        all_literals1 = env1.all_ground_literals
        self.all_literals1 = all_literals1

        pddl_env = pddlgym.make("PDDLEnvBlocks-v0")
        pddl_env.fix_problem_index(2)
        env2 = PDDLGymVecWrapper(pddl_env)
        all_literals2 = env2.all_ground_literals
        self.all_literals2 = all_literals2
        self.assertCountEqual(self.all_literals1, self.all_literals2)

    def test_PDDLEnvMinecraft_v0(self):
        pddl_env = pddlgym.make("PDDLEnvMinecraft-v0")
        pddl_env.fix_problem_index(2)
        env1 = PDDLGymVecWrapper(pddl_env)
        all_literals1 = env1.all_ground_literals
        self.all_literals1 = all_literals1

        pddl_env = pddlgym.make("PDDLEnvMinecraft-v0")
        pddl_env.fix_problem_index(2)
        env2 = PDDLGymVecWrapper(pddl_env)
        all_literals2 = env2.all_ground_literals
        self.all_literals2 = all_literals2
        self.assertCountEqual(self.all_literals1, self.all_literals2)


if __name__ == "__main__":

    unittest.main()
    exit()
