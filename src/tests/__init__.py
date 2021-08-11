import unittest
from tests.env_wrapper_test import TestListElements


def suite():
    suite = unittest.TestSuite()
    suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestListElements))
    return suite
