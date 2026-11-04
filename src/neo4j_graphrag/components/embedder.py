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
from typing import Optional

from pydantic import validate_call
import asyncio

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.components.types import TextChunk, TextChunks
from neo4j_graphrag.components.base import Component
from neo4j_graphrag.exceptions import EmbeddingsGenerationError


class TextChunkEmbedder(Component):
    """Component for creating embeddings from text chunks.

    Args:
        embedder (Embedder): The embedder to use to create the embeddings.
        max_concurrency (int): The maximum number of concurrent tasks which can be used to make requests to the embedder. Default is 5.
        batch_size (Optional[int]): If set, chunks are embedded in groups of up to
            ``batch_size`` texts through :meth:`Embedder.async_embed_documents`, so
            embedders that accept several inputs per request (OpenAI, Azure OpenAI,
            Cohere, Mistral AI, Ollama, SentenceTransformers) make one request per
            group instead of one per chunk. ``max_concurrency`` then limits the
            number of groups in flight. Default is None: one request per chunk.

    Example:

    .. code-block:: python

        from neo4j_graphrag.components.embedder import TextChunkEmbedder
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
        from neo4j_graphrag.experimental.pipeline import Pipeline

        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        chunk_embedder = TextChunkEmbedder(embedder)
        pipeline = Pipeline()
        pipeline.add_component(chunk_embedder, "chunk_embedder")

        # Embed 100 chunks per request instead of one request per chunk
        batched_chunk_embedder = TextChunkEmbedder(embedder, batch_size=100)

    """

    def __init__(
        self,
        embedder: Embedder,
        max_concurrency: int = 5,
        batch_size: Optional[int] = None,
    ):
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be a positive integer or None")
        self._embedder = embedder
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size

    def _with_embedding(
        self, text_chunk: TextChunk, embedding: list[float]
    ) -> TextChunk:
        metadata = text_chunk.metadata if text_chunk.metadata else {}
        metadata["embedding"] = embedding
        metadata["embedding_model_name"] = self._embedder.embedding_model_name
        return TextChunk(
            text=text_chunk.text,
            index=text_chunk.index,
            metadata=metadata,
            uid=text_chunk.uid,
        )

    def _embed_chunk(self, text_chunk: TextChunk) -> TextChunk:
        """Embed a single text chunk.

        Args:
            text_chunk (TextChunk): The text chunk to embed.

        Returns:
            TextChunk: The text chunk with an added "embedding" key in its
            metadata containing the embeddings of the text chunk's text.
        """
        embedding = self._embedder.embed_query(text_chunk.text)
        return self._with_embedding(text_chunk, embedding)

    async def _async_embed_chunk(
        self, sem: asyncio.Semaphore, text_chunk: TextChunk
    ) -> TextChunk:
        """Asynchronously embed a single text chunk.

        Args:
            text_chunk (TextChunk): The text chunk to embed.
            sem (asyncio.Semaphore): Semaphore to limit concurrency.

        Returns:
            TextChunk: The text chunk with an added "embedding" key in its
            metadata containing the embeddings of the text chunk's text.
        """
        async with sem:
            embedding = await self._embedder.async_embed_query(text_chunk.text)
            return self._with_embedding(text_chunk, embedding)

    async def _async_embed_batch(
        self, sem: asyncio.Semaphore, text_chunks: list[TextChunk]
    ) -> list[TextChunk]:
        """Asynchronously embed a group of text chunks with one
        ``async_embed_documents`` call.

        Args:
            sem (asyncio.Semaphore): Semaphore to limit concurrency.
            text_chunks (list[TextChunk]): The text chunks to embed.

        Returns:
            list[TextChunk]: The text chunks, in the same order, each with an
            added "embedding" key in its metadata.
        """
        async with sem:
            embeddings = await self._embedder.async_embed_documents(
                [text_chunk.text for text_chunk in text_chunks]
            )
        if len(embeddings) != len(text_chunks):
            raise EmbeddingsGenerationError(
                f"Embedder returned {len(embeddings)} embeddings "
                f"for {len(text_chunks)} chunks."
            )
        return [
            self._with_embedding(text_chunk, embedding)
            for text_chunk, embedding in zip(text_chunks, embeddings)
        ]

    @validate_call
    async def run(self, text_chunks: TextChunks) -> TextChunks:
        """Embed a list of text chunks.

        Args:
            text_chunks (TextChunks): The text chunks to embed.

        Returns:
            TextChunks: The input text chunks with each one having an added embedding.
        """
        sem = asyncio.Semaphore(self.max_concurrency)
        if self.batch_size is not None:
            size = self.batch_size
            groups = [
                text_chunks.chunks[i : i + size]
                for i in range(0, len(text_chunks.chunks), size)
            ]
            embedded_groups = await asyncio.gather(
                *(self._async_embed_batch(sem, group) for group in groups)
            )
            return TextChunks(
                chunks=[chunk for group in embedded_groups for chunk in group]
            )
        tasks = [
            self._async_embed_chunk(sem, text_chunk)
            for text_chunk in text_chunks.chunks
        ]
        chunks: list[TextChunk] = list(await asyncio.gather(*tasks))
        return TextChunks(chunks=chunks)
