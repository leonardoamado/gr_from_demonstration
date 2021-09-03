from typing import Collection, Tuple
from unittest.case import skip
from gym.core import Env
from pddlgym.structs import Literal
from ml.rl import TabularDynaQLearner, TabularPrioritisedQLearner, TabularQLearner, print_q_values
import ml.common
from recognizer import ActionQmaxRecognizer, Recognizer, StateQmaxRecognizer
from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD
from utils import solve_fset, find_action, find_actions
from ml.metrics import kl_divergence_norm_softmax, divergence_point, soft_divergence_point, trajectory_q_value
import numpy as np

import unittest


class RecognizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("*********Testing Recognizer Classes*********")

    def setUp(self):
        print("Unit tests for the recognizer")

    def train_recognizer_in_blocks_env(self, env: Env, recog: Recognizer):
        policies, actions = recog.train_policies(env)
        # Basic sanity check
        self.assertIsNotNone(policies)
        self.assertIsNotNone(actions)
        self.assertEquals(len(actions), 35)
        # TODO Check that all of the problems here really do start with "stack(e,c)"
        print(f'Initial State for 0 = {env.problems[0].initial_state}')
        print(f'Goal State for 0 = {env.problems[0].goal}')
        initial_state = solve_fset(env.problems[0].initial_state)
        action_index = policies[0].policy(initial_state)
        action = actions[action_index]
        print(f'Action 0={action_index}:{action}')
        print(f"Action for Initial State: {action}")
        print(f"Q-value: {policies[0].get_q_value(initial_state, action_index)}")
        print(print_q_values(policies[0].get_all_q_values(initial_state), actions))
        self.assertIn(action, find_actions(["unstack(c:block)", "stack(c:block,a:block)", "putdown(c:block)", "unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        print(f'Initial State for 1 = {env.problems[1].initial_state}')
        print(f'Goal State for 1 = {env.problems[1].goal}')
        initial_state = solve_fset(env.problems[1].initial_state)
        action_index = policies[1].policy(initial_state)
        action = actions[action_index]
        print(f'Action 1={action_index}:{action}')
        print(f"Action for Initial State: {action}")
        print(f"Q-value: {policies[0].get_q_value(initial_state, action_index)}")
        print(print_q_values(policies[0].get_all_q_values(initial_state), actions))
        self.assertIn(action, find_actions(["unstack(d:block)", "pickup(b:block)", "pickup(a:block)"], actions))
        print(f'Initial State for 2 = {env.problems[2].initial_state}')
        print(f'Goal State for 2 = {env.problems[2].goal}')
        initial_state = solve_fset(env.problems[2].initial_state)
        action_index = policies[2].policy(initial_state)
        action = actions[action_index]
        print(f'Action 0={action_index}:{action}')
        print(f"Action for Initial State: {action}")
        print(f"Q-value: {policies[2].get_q_value(initial_state, action_index)}")
        print(print_q_values(policies[2].get_all_q_values(initial_state), actions))
        self.assertIn(action, find_actions(["pickup(a:block)"], actions))
        print(actions)
        return policies, actions

    @skip
    def test_q_policy_learning(self):
        print("****  Testing Learning using TabularQLearning  ****")
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=False)
        recog = Recognizer()
        self.train_recognizer_in_blocks_env(env, recog)

    @skip  # This does not seem to do much better than TabularQLearning
    def test_dynaq_policy_learning(self):
        print("****  Testing Learning using TabularDynaQLearning  ****")
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=False)
        recog = Recognizer(method=TabularDynaQLearner)
        self.train_recognizer_in_blocks_env(env, recog)

    @skip
    def test_priority_policy_learning(self):
        print("****  Testing Learning using TabularPrioritisedQLearner  ****")
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=False)
        recog = Recognizer(method=TabularPrioritisedQLearner)
        self.train_recognizer_in_blocks_env(env, recog)

    def generate_observations(self, env: PDDLEnv, correct_goal_index: int) -> Collection[Tuple]:
        planner = FD()
        env.fix_problem_index(correct_goal_index)
        init, _ = env.reset()
        plan = planner(env.domain, init)
        traj = []
        for a in plan:
            state_action_pair = (solve_fset(init.literals), a)
            traj.append(state_action_pair)
            init, _, _, _ = env.step(a)
        return traj

    @skip
    def test_state_action_recognizer_rl_algorithms_blocks(self):
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        for rlalgorithm in [TabularQLearner, TabularDynaQLearner, TabularPrioritisedQLearner]:
            print(f"Testing Recognizer using {rlalgorithm} training")
            recog = Recognizer(method=rlalgorithm)
            policies, actions = recog.train_policies(env)
            correct_goal_index = 1
            traj = self.generate_observations(env, correct_goal_index)
            print(f"Observations are {traj}")
            success, goal, rankings = recog.recognize_process(env, policies, actions, traj, real_goal=correct_goal_index, n_goals=3)
            self.assertEqual(goal, correct_goal_index)
            self.assertTrue(success)
            self.assertIsNotNone(rankings)
            print(rankings)
            self.assertEqual(correct_goal_index, np.argmin(np.transpose(rankings)[1]))

    def test_state_action_recognizer_evaluation(self):
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        training_recognizer = Recognizer()
        policies, actions = training_recognizer.train_policies(env)
        for evaluation in [kl_divergence_norm_softmax, trajectory_q_value, soft_divergence_point  # , 
                           # divergence_point  # divergence point is not brilliant
                           ]:
            print(f"Testing Recognizer using {evaluation} evaluation")
            recog = Recognizer(evaluation=evaluation)
            correct_goal_index = 1
            traj = self.generate_observations(env, correct_goal_index)
            print(f"Observations are {traj}")
            success, goal, rankings = recog.recognize_process(env, policies, actions, traj, real_goal=correct_goal_index, n_goals=3)
            self.assertEqual(goal, correct_goal_index)
            self.assertTrue(success)
            self.assertIsNotNone(rankings)
            print(rankings)
            self.assertEqual(correct_goal_index, np.argmin(np.transpose(rankings)[1]))

    @skip
    def test_action_recognition_blocks(self):
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog = StateQmaxRecognizer()
        policies, actions = recog.train_policies(env)
        correct_goal_index = 1
        traj = self.generate_observations(env, correct_goal_index)
        print(f"Observations are {traj}")
        success, goal, rankings = recog.recognize_process(env, policies, actions, traj, real_goal=correct_goal_index, n_goals=3)
        self.assertEqual(goal, correct_goal_index)
        self.assertTrue(success)
        self.assertIsNotNone(rankings)
        print(rankings)
        self.assertEqual(correct_goal_index, np.argmax(np.transpose(rankings)[1]))

    @skip
    def test_state_recognition_blocks(self):
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog = ActionQmaxRecognizer()
        policies, actions = recog.train_policies(env)
        correct_goal_index = 1
        traj = self.generate_observations(env, correct_goal_index)
        print(f"Observations are {traj}")
        success, goal, rankings = recog.recognize_process(env, policies, actions, traj, real_goal=correct_goal_index, n_goals=3)
        print(rankings)
        self.assertEqual(goal, correct_goal_index)
        self.assertTrue(success)
        self.assertIsNotNone(rankings)
        self.assertEqual(correct_goal_index, np.argmax(np.transpose(rankings)[1]))

    @skip  # This does not work because of various dependencies
    def test_parallel_training(self):
        env = PDDLEnv(ml.common.ROOT_DIR+'/output/blocks_gr/blocks_gr.pddl', ml.common.ROOT_DIR+'/output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog1 = Recognizer()
        policies1, actions1 = recog1.train_policies(env)
        recog2 = Recognizer()
        policies2, actions2 = recog2.train_parallel(env)
        self.assertEqual(actions1, actions2)
        for i, policy in enumerate(policies1):
            initial_state, _ = solve_fset(env.problems[i].reset())
            self.assertEqual(policy.policy(initial_state), policies2[i].policy(initial_state))


if __name__ == "__main__":
    unittest.main()
    exit()
