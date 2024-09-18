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

from neo4j_graphrag.experimental.components.text_splitters.base import TextSplitter
from neo4j_graphrag.experimental.components.types import TextChunk, TextChunks


class FixedSizeSplitter(TextSplitter):
    """Text splitter which splits the input text into fixed size chunks with optional overlap.

    Args:
        chunk_size (int): The number of characters in each chunk.
        chunk_overlap (int): The number of characters from the previous chunk to overlap with each chunk. Must be less than `chunk_size`.

    Example:

    .. code-block:: python

        from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
        from neo4j_graphrag.experimental.pipeline import Pipeline

        pipeline = Pipeline()
        text_splitter = FixedSizeSplitter(chunk_size=4000, chunk_overlap=200)
        pipeline.add_component(text_splitter, "text_splitter")
    """

    @validate_call
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @validate_call
    async def run(self, text: str) -> TextChunks:
        """Splits a piece of text into chunks.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of chunks.
        """
        chunks = []
        index = 0
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            start = i
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunks.append(TextChunk(text=chunk_text, index=index))
            index += 1
        return TextChunks(chunks=chunks)
