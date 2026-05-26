import pandas as pd
import pytest

from unittest.mock import MagicMock
import challenge.api as api_module


# API tests verify HTTP behavior only (routing, validation, response shape).
# Model correctness is covered by model tests. A mock avoids training overhead
# and keeps the test boundary clean — the API layer is tested in isolation.
@pytest.fixture(scope="session", autouse=True)
def mock_model():
    mock = MagicMock()
    mock.preprocess.return_value = pd.DataFrame()
    mock.predict.return_value = [0]
    api_module.model = mock
