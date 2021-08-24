from typing import Collection
from unittest.case import skip
from pddlgym.structs import Literal
from ml.rl import TabularDynaQLearner
from recognizer import Recognizer
from pddlgym.core import PDDLEnv

import unittest


def find_action(action_name: str, actions: Collection[Literal]) -> Literal:
    for action in actions:
        if action.__repr__() == action_name:
            return action
    return None


class RecognizerTest(unittest.TestCase):
    def setUp(self):
        print("Unit tests for the recognizer")

    def test_q_policy_learning(self):
        print("****  Testing Learning using TabularQLearning  ****")
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=False)
        recog = Recognizer()
        policies, actions = recog.train_policies(env)
        # Basic sanity check
        self.assertIsNotNone(policies)
        self.assertIsNotNone(actions)
        self.assertEquals(len(actions), 35)
        # TODO Check that all of the problems here really do start with "stack(e,c)"
        action = actions[policies[0].agent_step(0.0, env.problems[0].initial_state)]
        print(f'Action 0={policies[0].agent_step(0.0, env.problems[0].initial_state)}:{action}')
        self.assertEqual(find_action("unstack(d:block)", actions), action)
        action = actions[policies[1].agent_step(0.0, env.problems[1].initial_state)]
        print(f'Action 1={policies[1].agent_step(0.0, env.problems[1].initial_state)}:{action}')
        self.assertEqual(find_action("unstack(d:block)", actions), action)
        action = actions[policies[2].agent_step(0.0, env.problems[2].initial_state)]
        print(f'Action 2={policies[2].agent_step(0.0, env.problems[2].initial_state)}:{action}')
        self.assertEqual(find_action("pickup(a:block)", actions), action)
        print(actions)

    @skip  # This does not seem to do much better than TabularQLearning
    def test_dynaq_policy_learning(self):
        print("****  Testing Learning using TabularDynaQLearning  ****")
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=False)
        recog = Recognizer(method=TabularDynaQLearner)
        policies, actions = recog.train_policies(env)
        # Basic sanity check
        self.assertIsNotNone(policies)
        self.assertIsNotNone(actions)
        self.assertEquals(len(actions), 35)
        # TODO Check that all of the problems here really do start with "stack(e,c)"
        action = actions[policies[0].agent_step(0.0, env.problems[0].initial_state)]
        print(f'Action 0={policies[0].agent_step(0.0, env.problems[0].initial_state)}:{action}')
        # self.assertEqual(find_action("unstack(d:block)", actions), action)
        action = actions[policies[1].agent_step(0.0, env.problems[1].initial_state)]
        print(f'Action 1={policies[1].agent_step(0.0, env.problems[1].initial_state)}:{action}')
        # self.assertEqual(find_action("stack(e:block,c:block)", actions), action)
        action = actions[policies[2].agent_step(0.0, env.problems[2].initial_state)]
        print(f'Action 2={policies[2].agent_step(0.0, env.problems[2].initial_state)}:{action}')
        # self.assertEqual(find_action("stack(e:block,c:block)", actions), action)
        print(actions)

    def test_blocks(self):
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog = Recognizer()


if __name__ == "__main__":
    unittest.main()
    exit()
