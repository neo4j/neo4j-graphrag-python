import pytest
from neo4j import GraphDatabase
from neo4j_genai_python.src import GenAIClient, Embeddings

@pytest.fixture
def genai_client(mocker):
    mock_driver = mocker.MagicMock(spec=GraphDatabase.driver)
    mock_embeddings = mocker.MagicMock(spec=Embeddings)
    client = GenAIClient(driver=mock_driver, embeddings=mock_embeddings)
    return client