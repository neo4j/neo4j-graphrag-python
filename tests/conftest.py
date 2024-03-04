import pytest
from neo4j_genai import GenAIClient
from unittest.mock import Mock, patch


@pytest.fixture
def driver():
    return Mock()


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client(_verify_version_mock, driver):
    return GenAIClient(driver)
