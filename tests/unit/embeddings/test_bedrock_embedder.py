#  Copyright (c) "Neo4j"
#  Neo4j Sweden AB [https://neo4j.com]
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#      https://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from __future__ import annotations

import io
import json
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from neo4j_graphrag.embeddings.bedrock import BedrockEmbeddings
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@pytest.fixture
def mock_boto3() -> Generator[MagicMock, None, None]:
    with patch("neo4j_graphrag.embeddings.bedrock.boto3") as mock_boto:
        mock_client = MagicMock()
        mock_boto.client.return_value = mock_client
        yield mock_boto


def _make_invoke_response(embedding: list[float]) -> dict:
    body_bytes = json.dumps({"embedding": embedding}).encode()
    return {"body": io.BytesIO(body_bytes)}


@patch("neo4j_graphrag.embeddings.bedrock.boto3", None)
def test_bedrock_embedder_missing_dependency() -> None:
    with pytest.raises(ImportError) as exc:
        BedrockEmbeddings()
    assert "Could not import boto3 python client" in str(exc.value)


def test_bedrock_embed_query_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.invoke_model.return_value = _make_invoke_response([0.1, 0.2, 0.3])

    embedder = BedrockEmbeddings()
    res = embedder.embed_query("hello")

    assert res == [0.1, 0.2, 0.3]
    mock_client.invoke_model.assert_called_once()
    call_kwargs = mock_client.invoke_model.call_args[1]
    body = json.loads(call_kwargs["body"])
    assert body["inputText"] == "hello"
    assert body["dimensions"] == 1024
    assert body["normalize"] is True


@pytest.mark.asyncio
async def test_bedrock_async_embed_query_happy_path(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.invoke_model.return_value = _make_invoke_response([0.4, 0.5, 0.6])

    embedder = BedrockEmbeddings()
    res = await embedder.async_embed_query("hello")

    assert res == [0.4, 0.5, 0.6]
    mock_client.invoke_model.assert_called_once()


def test_bedrock_embed_query_error(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.invoke_model.side_effect = Exception("API error")

    embedder = BedrockEmbeddings()
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with Bedrock"
    ):
        embedder.embed_query("hello")

    assert mock_client.invoke_model.call_count == 1


def test_bedrock_embed_query_custom_params(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    mock_client.invoke_model.return_value = _make_invoke_response([1.0, 2.0])

    embedder = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v1",
        dimensions=512,
        normalize=False,
        region_name="eu-west-1",
    )
    res = embedder.embed_query("test")

    assert res == [1.0, 2.0]
    call_kwargs = mock_client.invoke_model.call_args[1]
    assert call_kwargs["modelId"] == "amazon.titan-embed-text-v1"
    body = json.loads(call_kwargs["body"])
    assert body["dimensions"] == 512
    assert body["normalize"] is False


def test_bedrock_embed_query_empty_response(mock_boto3: MagicMock) -> None:
    mock_client = mock_boto3.client.return_value
    body_bytes = json.dumps({"embedding": None}).encode()
    mock_client.invoke_model.return_value = {"body": io.BytesIO(body_bytes)}

    embedder = BedrockEmbeddings()
    with pytest.raises(
        EmbeddingsGenerationError, match="Failed to generate embedding with Bedrock"
    ):
        embedder.embed_query("hello")
