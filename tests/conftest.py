import pytest
from neo4j_genai import GenAIClient
from unittest.mock import Mock, patch
from typing import List
from neo4j_genai.embedder import Embedder


@pytest.fixture
def driver():
    return Mock()


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client(_verify_version_mock, driver):
    return GenAIClient(driver)


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client_with_embedder(_verify_version_mock, driver):
    class CustomEmbedder(Embedder):
        def embed_query(self, text: str) -> List[float]:
            return [1.0 for _ in range(1536)]

    embedder = CustomEmbedder()
    return GenAIClient(driver, embedder)
