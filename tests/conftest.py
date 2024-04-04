import pytest
from neo4j_genai import VectorRetriever
from neo4j import Driver
from unittest.mock import MagicMock, patch


@pytest.fixture
def driver():
    return MagicMock(spec=Driver)


@pytest.fixture
@patch("neo4j_genai.VectorRetriever._verify_version")
def retriever(_verify_version_mock, driver):
    return VectorRetriever(driver)
