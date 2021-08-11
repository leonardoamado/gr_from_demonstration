from tests import suite
import unittest

if __name__ == "__main__":
    runner = unittest.TextTestRunner()
    test_suite = suite()
    runner.run(test_suite)
