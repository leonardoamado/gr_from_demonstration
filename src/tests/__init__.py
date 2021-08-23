import unittest
from tests.env_wrapper_test import TestListElements
from tests.recognizer_test import RecognizerTest


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestListElements))
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(RecognizerTest))
    return suite
