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
from pydantic import validate_call

from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks
from neo4j_graphrag.experimental.pipeline.component import Component


class TextChunkEmbedder(Component):
    """Component for creating embeddings from text chunks.

    Args:
        embedder (Embedder): The embedder to use to create the embeddings.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.embedder import TextChunkEmbedder
        from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
        from neo4j_graphrag.experimental.pipeline import Pipeline

        embedder = OpenAIEmbeddings(model="text-embedding-3-large")
        chunk_embedder = TextChunkEmbedder(embedder)
        pipeline = Pipeline()
        pipeline.add_component(chunk_embedder, "chunk_embedder")

    """

    def __init__(self, embedder: Embedder):
        self._embedder = embedder

    def _embed_chunk(self, text_chunk: TextChunk) -> TextChunk:
        """Embed a single text chunk.

        Args:
            text_chunk (TextChunk): The text chunk to embed.

        Returns:
            TextChunk: The text chunk with an added "embedding" key in its
            metadata containing the embeddings of the text chunk's text.
        """
        embedding = self._embedder.embed_query(text_chunk.text)
        metadata = text_chunk.metadata if text_chunk.metadata else {}
        metadata["embedding"] = embedding
        return TextChunk(
            text=text_chunk.text, index=text_chunk.index, metadata=metadata
        )

    @validate_call
    async def run(self, text_chunks: TextChunks) -> TextChunks:
        """Embed a list of text chunks.

        Args:
            text_chunks (TextChunks): The text chunks to embed.

        Returns:
            TextChunks: The input text chunks with each one having an added embedding.
        """
        return TextChunks(
            chunks=[self._embed_chunk(text_chunk) for text_chunk in text_chunks.chunks]
        )
