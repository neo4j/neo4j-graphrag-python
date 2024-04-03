import pytest
from neo4j_genai import GenAIClient
from neo4j import Driver
from unittest.mock import MagicMock, patch


@pytest.fixture
def driver():
    return MagicMock(spec=Driver)


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client(_verify_version_mock, driver):
    return GenAIClient(driver)
