import random
from random import Random
import unittest
from unittest.case import skip

from pddlgym.core import PDDLEnv

from ml.rl import RLAgent, TabularDynaQLearner, TabularPrioritisedQLearner, TabularQLearner

from utils import find_action, find_actions, solve_fset

import ml.common


class RLAgentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("*********Testing RLAgent*********")

    # @skip
    def test_tabular_q(self):
        random.seed(1)
        print(f"ROOT_DIR is {ml.common.ROOT_DIR}")
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        env.fix_problem_index(0)
        env.seed(9)  # fixing seed to ensure test is reproducible
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularQLearner(env, init, action_list=actions, episodes=10000, rand=Random(1))
        agent.learn()
        print(actions)
        initial_state = solve_fset(env.problems[0].initial_state)
        policy_index = agent.policy(initial_state)
        policy_action = actions[policy_index]
        self.assertIn(policy_action, find_actions(["unstack(d:block)", "pickup(a:block)"], actions))
        print(policy_action)
        print(agent.get_q_value(initial_state, policy_index))
        self.assertGreater(agent.get_q_value(initial_state, policy_index), 48.)

    # @skip
    def test_tabular_dyna_q(self):
        random.seed(1)
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        env.fix_problem_index(0)
        env.seed(9)  # fixing seed to ensure test is reproducible
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularDynaQLearner(env, init, action_list=actions, episodes=10000, rand=Random(1))
        agent.learn()
        print(actions)
        initial_state = solve_fset(env.problems[0].initial_state)
        policy_index = agent.policy(initial_state)
        policy_action = actions[policy_index]
        self.assertIn(policy_action, find_actions(["unstack(d:block)", "pickup(a:block)"], actions))
        print(policy_action)
        print(agent.get_q_value(initial_state, policy_index))
        self.assertGreater(agent.get_q_value(initial_state, policy_index), 48.)

    def test_prioritised_dyna_q(self):
        random.seed(1)
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', True, False)
        # get all literals in its grounded version
        env.fix_problem_index(0)
        env.seed(9)  # fixing seed to ensure test is reproducible
        init, _ = env.reset()
        actions = list(env.action_space.all_ground_literals(init, valid_only=False))
        agent = TabularPrioritisedQLearner(env, init, action_list=actions, episodes=10000, rand=Random(1))
        agent.learn()
        print(actions)
        initial_state = solve_fset(env.problems[0].initial_state)
        policy_index = agent.policy(initial_state)
        policy_action = actions[policy_index]
        self.assertIn(policy_action, find_actions(["unstack(d:block)", "pickup(a:block)"], actions))
        print(policy_action)
        print(agent.get_q_value(initial_state, policy_index))
        self.assertGreater(agent.get_q_value(initial_state, policy_index), 48.)


if __name__ == "__main__":
    unittest.main()
    exit()
