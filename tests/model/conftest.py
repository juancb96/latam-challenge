import os
import pytest


# test_model.py loads data with the hardcoded relative path "../data/data.csv".
# pytest runs from the project root (challenge_MLE/), so that path resolves to
# the parent directory — which doesn't contain data/. Changing cwd to
# challenge_MLE/challenge/ before each test makes "../data/data.csv" resolve
# correctly to challenge_MLE/data/data.csv without modifying the test file.
@pytest.fixture(autouse=True)
def set_working_directory():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    orig = os.getcwd()
    os.chdir(os.path.join(project_root, 'challenge'))
    yield
    os.chdir(orig)
