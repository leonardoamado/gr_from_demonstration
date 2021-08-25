from typing import Collection
from unittest.case import skip
from pddlgym.structs import Literal
from ml.rl import TabularDynaQLearner
from recognizer import Recognizer
from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD

import unittest


def find_action(action_name: str, actions: Collection[Literal]) -> Literal:
    for action in actions:
        if action.__repr__() == action_name:
            return action
    return None


def find_actions(action_names: Collection[str], actions: Collection[Literal]) -> Collection[Literal]:
    literals = []
    for action in actions:
        if action.__repr__() in action_names:
            literals.append(action)
    return literals


class RecognizerTest(unittest.TestCase):
    def setUp(self):
        print("Unit tests for the recognizer")

    @skip
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
        print(f'Initial State for 0 = {env.problems[0].initial_state}')
        print(f'Goal State for 0 = {env.problems[0].goal}')
        action = actions[policies[0].agent_step(0.0, env.problems[0].initial_state)]
        print(f'Action 0={policies[0].agent_step(0.0, env.problems[0].initial_state)}:{action}')
        self.assertIn(action, find_actions(["unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        print(f'Initial State for 1 = {env.problems[1].initial_state}')
        print(f'Goal State for 1 = {env.problems[1].goal}')
        action = actions[policies[1].agent_step(0.0, env.problems[1].initial_state)]
        print(f'Action 1={policies[1].agent_step(0.0, env.problems[1].initial_state)}:{action}')
        self.assertIn(action, find_actions(["unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        print(f'Initial State for 2 = {env.problems[2].initial_state}')
        print(f'Goal State for 2 = {env.problems[2].goal}')
        action = actions[policies[2].agent_step(0.0, env.problems[2].initial_state)]
        print(f'Action 2={policies[2].agent_step(0.0, env.problems[2].initial_state)}:{action}')
        self.assertIn(action, find_actions(["pickup(a:block)"], actions))
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
        self.assertIn(action, find_actions(["unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        action = actions[policies[1].agent_step(0.0, env.problems[1].initial_state)]
        print(f'Action 1={policies[1].agent_step(0.0, env.problems[1].initial_state)}:{action}')
        self.assertIn(action, find_actions(["unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        action = actions[policies[2].agent_step(0.0, env.problems[2].initial_state)]
        print(f'Action 2={policies[2].agent_step(0.0, env.problems[2].initial_state)}:{action}')
        self.assertIn(action, find_actions(["pickup(a:block)"], actions))
        print(actions)

    def test_blocks(self):
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog = Recognizer()
        policies, actions = recog.train_policies(env)
        planner = FD()
        correct_goal_index = 1
        env.fix_problem_index(correct_goal_index)
        plan = planner(env.domain, env.reset()[0])
        print(f"Observations are {plan}")
        success, goal, rankings = recog.recognize_goal(env, policies, actions, plan, real_goal=correct_goal_index, n_goals=3)
        self.assertEqual(goal, correct_goal_index)
        self.assertTrue(success)
        self.assertIsNotNone(rankings)
        self.assertEquals(correct_goal_index, rankings[0])


if __name__ == "__main__":
    unittest.main()
    exit()
