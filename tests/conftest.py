import pytest
from neo4j_genai import GenAIClient
from unittest.mock import MagicMock, patch
from typing import List
from neo4j_genai.embedder import Embedder


@pytest.fixture
def driver():
    return MagicMock()


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client(_verify_version_mock, driver):
    return GenAIClient(driver)
