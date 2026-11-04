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
import asyncio
from unittest.mock import MagicMock

import pytest
from neo4j_graphrag.components.embedder import TextChunkEmbedder
from neo4j_graphrag.components.types import (
    TextChunk,
    TextChunks,
)
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


@pytest.mark.asyncio
async def test_text_chunk_embedder_run(embedder: MagicMock) -> None:
    embedder.async_embed_query.return_value = [1.0, 2.0, 3.0]
    embedder.embedding_model_name = "my_embedder"
    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    text_chunks = TextChunks(
        chunks=[TextChunk(text="may thy knife chip and shatter", index=0)]
    )
    embedded_chunks = await text_chunk_embedder.run(text_chunks)
    embedder.async_embed_query.assert_called_once_with("may thy knife chip and shatter")
    assert isinstance(embedded_chunks, TextChunks)
    for chunk in embedded_chunks.chunks:
        assert isinstance(chunk, TextChunk)
        assert chunk.metadata is not None
        assert "embedding" in chunk.metadata.keys()
        assert isinstance(chunk.metadata["embedding"], list)
        for i in chunk.metadata["embedding"]:
            assert isinstance(i, float)
        assert chunk.metadata["embedding_model_name"] == "my_embedder"


def _chunks(n: int) -> TextChunks:
    return TextChunks(chunks=[TextChunk(text=f"text {i}", index=i) for i in range(n)])


@pytest.mark.asyncio
async def test_text_chunk_embedder_batched_run(embedder: MagicMock) -> None:
    async def fake_embed_documents(texts: list[str]) -> list[list[float]]:
        return [[float(t.split()[1])] for t in texts]

    embedder.async_embed_documents.side_effect = fake_embed_documents
    embedder.embedding_model_name = "my_embedder"
    text_chunk_embedder = TextChunkEmbedder(embedder=embedder, batch_size=2)

    result = await text_chunk_embedder.run(_chunks(5))

    assert [c.args[0] for c in embedder.async_embed_documents.call_args_list] == [
        ["text 0", "text 1"],
        ["text 2", "text 3"],
        ["text 4"],
    ]
    embedder.async_embed_query.assert_not_called()
    assert [c.index for c in result.chunks] == [0, 1, 2, 3, 4]
    for chunk in result.chunks:
        assert chunk.metadata is not None
        assert chunk.metadata["embedding"] == [float(chunk.index)]
        assert chunk.metadata["embedding_model_name"] == "my_embedder"


@pytest.mark.asyncio
async def test_text_chunk_embedder_batched_keeps_metadata_and_uid(
    embedder: MagicMock,
) -> None:
    embedder.async_embed_documents.return_value = [[1.0]]
    embedder.embedding_model_name = "m"
    chunk = TextChunk(text="t", index=0, metadata={"page": 3})
    result = await TextChunkEmbedder(embedder, batch_size=10).run(
        TextChunks(chunks=[chunk])
    )
    assert result.chunks[0].uid == chunk.uid
    assert result.chunks[0].metadata == {
        "page": 3,
        "embedding": [1.0],
        "embedding_model_name": "m",
    }


@pytest.mark.asyncio
async def test_text_chunk_embedder_batched_respects_max_concurrency(
    embedder: MagicMock,
) -> None:
    in_flight = 0
    peak = 0

    async def slow_embed_documents(texts: list[str]) -> list[list[float]]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return [[0.0] for _ in texts]

    embedder.async_embed_documents.side_effect = slow_embed_documents
    embedder.embedding_model_name = ""
    await TextChunkEmbedder(embedder, max_concurrency=2, batch_size=1).run(_chunks(6))
    assert embedder.async_embed_documents.await_count == 6
    assert peak == 2


@pytest.mark.asyncio
async def test_text_chunk_embedder_batched_empty(embedder: MagicMock) -> None:
    result = await TextChunkEmbedder(embedder, batch_size=4).run(TextChunks(chunks=[]))
    assert result.chunks == []
    embedder.async_embed_documents.assert_not_called()


@pytest.mark.asyncio
async def test_text_chunk_embedder_batched_count_mismatch(embedder: MagicMock) -> None:
    embedder.async_embed_documents.return_value = [[1.0]]
    with pytest.raises(EmbeddingsGenerationError, match="1 embeddings for 2 chunks"):
        await TextChunkEmbedder(embedder, batch_size=2).run(_chunks(2))


@pytest.mark.parametrize("batch_size", [0, -1])
def test_text_chunk_embedder_invalid_batch_size(
    embedder: MagicMock, batch_size: int
) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        TextChunkEmbedder(embedder, batch_size=batch_size)
