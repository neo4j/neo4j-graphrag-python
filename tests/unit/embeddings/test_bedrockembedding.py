from unittest.mock import patch, MagicMock
import pytest
import json

from neo4j_graphrag.embeddings.bedrockembeddings import BedrockEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@patch("neo4j_graphrag.embeddings.bedrockembeddings.boto3.client")
def test_bedrock_embedder_happy_path(mock_boto_client):
    # Mock AWS response with valid embedding
    fake_embedding = [0.1] * 1024
    fake_response = {
        "embedding": fake_embedding
    }

    # Mock the .read() to return the fake response as JSON bytes
    mock_body = MagicMock()
    mock_body.read.return_value = json.dumps(fake_response).encode("utf-8")

    # Mock the bedrock client
    mock_bedrock_client = MagicMock()
    mock_bedrock_client.invoke_model.return_value = {"body": mock_body}
    mock_boto_client.return_value = mock_bedrock_client

    # Instantiate the embedder and run embed_query
    embedder = BedrockEmbeddings()
    result = embedder.embed_query("Hello, Bedrock!")

    # Assertions
    assert isinstance(result, list)
    assert len(result) == 1024
    assert result == fake_embedding


@patch("neo4j_graphrag.embeddings.bedrockembeddings.boto3.client")
def test_bedrock_embedder_error_path(mock_boto_client):
    # Simulate AWS client raising an exception
    mock_bedrock_client = MagicMock()
    mock_bedrock_client.invoke_model.side_effect = Exception("AWS error")
    mock_boto_client.return_value = mock_bedrock_client

    embedder = BedrockEmbeddings()

    with pytest.raises(EmbeddingsGenerationError) as exc_info:
        embedder.embed_query("This will fail.")

    assert "Issue Generating Embeddings" in str(exc_info.value)
