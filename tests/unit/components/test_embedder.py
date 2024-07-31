from unittest.mock import MagicMock

import pytest
from neo4j_genai.components.embedder import TextChunkEmbedder
from neo4j_genai.components.types import TextChunk, TextChunks


@pytest.mark.asyncio
async def test_text_chunk_embedder_run(embedder: MagicMock):
    embedder.embed_query.return_value = [1.0, 2.0, 3.0]
    text_chunk_embedder = TextChunkEmbedder(embedder=embedder)
    text_chunks = TextChunks(chunks=[TextChunk(text="may thy knife chip and shatter")])
    embedded_chunks = await text_chunk_embedder.run(text_chunks)
    embedder.embed_query.assert_called_once_with("may thy knife chip and shatter")
    assert isinstance(embedded_chunks, TextChunks)
    for chunk in embedded_chunks.chunks:
        assert isinstance(chunk, TextChunk)
        assert "embedding" in chunk.metadata.keys()
        assert isinstance(chunk.metadata["embedding"], list)
        for i in chunk.metadata["embedding"]:
            assert isinstance(i, float)
