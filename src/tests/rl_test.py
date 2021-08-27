from random import Random
import unittest

from pddlgym.core import PDDLEnv

from ml.rl import RLAgent, TabularQLearner


class RLAgentTest(unittest.TestCase):

    def test_tabular_q(self):
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularQLearner(env, init, action_list=actions, rand=Random(1))
        agent.learn()
        self.assertEqual(agent.policy(), 0)


if __name__ == "__main__":
    unittest.main()
    exit()
