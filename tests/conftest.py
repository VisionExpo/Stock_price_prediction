import pytest

@pytest.fixture(scope='session', autouse=True)
def setup_session():
    pass

@pytest.fixture(autouse=True)
def setup_function():
    pass