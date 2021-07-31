import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--slow", action="store_true", default=False, help="Also run slow tests"
    )


def pytest_runtest_setup(item):
    """Skip tests if they are marked as slow and --slow is not given"""
    if "slow" in item.keywords and not item.config.getvalue("slow"):
        pytest.skip("slow tests not requested")
