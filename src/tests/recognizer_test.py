from recognizer import Recognizer
from pddlgym.core import PDDLEnv

import unittest


class RecognizerTest(unittest.TestCase):
    def setUp(self):
        print("Unit tests for the recognizer")

    def test_policy_learning(self):
        env = PDDLEnv('output/blocks_gr/blocks_gr.pddl', 'output/blocks_gr/problems/', raise_error_on_invalid_action=True, dynamic_action_space=True)
        recog = Recognizer()
        policies, actions = recog.train_policies(env)
        # Basic sanity check
        self.assertIsNotNone(policies)
        self.assertIsNotNone(actions)
    
    def test_blocks(self):
        pass


if __name__ == "__main__":
    unittest.main()
    exit()
