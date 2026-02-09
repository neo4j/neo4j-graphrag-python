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

from typing import Generator, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from neo4j_graphrag.exceptions import EmbeddingsGenerationError
from neo4j_graphrag.embeddings import GeminiEmbedder


@pytest.fixture
def mock_genai() -> Generator[Tuple[MagicMock, MagicMock], None, None]:
    with patch("neo4j_graphrag.embeddings.google_genai.genai") as mock_genai, patch(
        "neo4j_graphrag.embeddings.google_genai.types"
    ) as mock_types:
        mock_client = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_types.EmbedContentConfig = MagicMock(side_effect=lambda **kwargs: kwargs)

        yield mock_genai, mock_types


def test_gemini_embedder_missing_dependency() -> None:
    with patch("neo4j_graphrag.embeddings.google_genai.genai", None):
        with pytest.raises(ImportError) as exc:
            GeminiEmbedder()
        assert "Could not import google-genai python client" in str(exc.value)


def test_gemini_embed_query_happy_path(mock_genai: Tuple[MagicMock, MagicMock]) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_result = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.1, 0.2, 0.3]
    mock_result.embeddings = [mock_embedding]
    mock_client.models.embed_content.return_value = mock_result

    embedder = GeminiEmbedder()
    res = embedder.embed_query("hello")

    assert res == [0.1, 0.2, 0.3]
    mock_client.models.embed_content.assert_called_once()


@pytest.mark.asyncio
async def test_gemini_async_embed_query_happy_path(
    mock_genai: Tuple[MagicMock, MagicMock],
) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_result = MagicMock()
    mock_embedding = MagicMock()
    mock_embedding.values = [0.4, 0.5, 0.6]
    mock_result.embeddings = [mock_embedding]
    mock_client.aio.models.embed_content = AsyncMock(return_value=mock_result)

    embedder = GeminiEmbedder()
    res = await embedder.async_embed_query("hello")

    assert res == [0.4, 0.5, 0.6]
    mock_client.aio.models.embed_content.assert_awaited_once()


def test_gemini_embed_query_error(mock_genai: Tuple[MagicMock, MagicMock]) -> None:
    mock_gen, _ = mock_genai
    mock_client = mock_gen.Client.return_value
    mock_client.models.embed_content.side_effect = Exception("API error")

    embedder = GeminiEmbedder()
    with pytest.raises(EmbeddingsGenerationError):
        embedder.embed_query("hello")
