import unittest
from tests.env_wrapper_test import TestListElements
from tests.recognizer_test import RecognizerTest
from tests.rl_test import RLAgentTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestListElements))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(RecognizerTest))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(RLAgentTest))
    return suite
