import random
from random import Random
import unittest

from pddlgym.core import PDDLEnv

from ml.rl import RLAgent, TabularDynaQLearner, TabularQLearner

from utils import find_action


class RLAgentTest(unittest.TestCase):

    def test_tabular_q(self):
        random.seed(1)
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        env.fix_problem_index(0)
        env.seed(9)  # fixing seed to ensure test is reproducible
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularQLearner(env, init, action_list=actions, episodes=10000, rand=Random(1))
        agent.learn()
        print(actions)
        policy_action = actions[agent.policy(env.problems[0].initial_state)]
        self.assertEqual(policy_action, find_action("pickup(a:block)", actions))

    def test_tabular_dyna_q(self):
        random.seed(1)
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        env.fix_problem_index(0)
        env.seed(9)  # fixing seed to ensure test is reproducible
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularDynaQLearner(env, init, action_list=actions, episodes=10000, rand=Random(1))
        agent.learn()
        print(actions)
        policy_action = actions[agent.policy(env.problems[0].initial_state)]
        self.assertEqual(policy_action, find_action("pickup(a:block)", actions))


if __name__ == "__main__":
    unittest.main()
    exit()
