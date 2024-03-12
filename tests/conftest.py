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


@pytest.fixture
@patch("neo4j_genai.GenAIClient._verify_version")
def client_with_embedder(_verify_version_mock, driver):
    class CustomEmbedder(Embedder):
        def __init__(self):
            self.dimension = 1536

        def embed_query(self, text: str) -> List[float]:
            return [1.0 for _ in range(self.dimension)]

        def set_dimension(self, dimension: int):
            self.dimension = dimension

    embedder = CustomEmbedder()
    return GenAIClient(driver, embedder), embedder
