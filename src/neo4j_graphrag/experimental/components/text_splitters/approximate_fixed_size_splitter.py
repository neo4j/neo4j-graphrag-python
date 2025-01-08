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


def _adjust_chunk_start(text: str, proposed_start: int) -> int:
    """
    Shift the starting index backward if it lands in the middle of a word.
    If no whitespace is found, use the proposed start.

     Args:
        text (str): The text being split.
        proposed_start (int): The initial starting index of the chunk.

    Returns:
        int: The adjusted starting index, ensuring the chunk does not begin in the
             middle of a word.
    """
    start = proposed_start
    if start > 0 and not text[start].isspace() and not text[start - 1].isspace():
        while start > 0 and not text[start - 1].isspace():
            start -= 1

        # fallback if no whitespace is found
        if start == 0 and not text[0].isspace():
            start = proposed_start
    return start


def _adjust_chunk_end(text: str, start: int, approximate_end: int) -> int:
    """
    Shift the ending index backward if it lands in the middle of a word.
    If no whitespace is found, use 'approximate_end' to avoid an infinite loop.

    Args:
        text (str): The full text being split.
        start (int): The adjusted starting index for this chunk.
        approximate_end (int): The initial end index.

    Returns:
        int: The adjusted ending index, ensuring the chunk does not end in the middle of
            a word if possible.
    """
    end = approximate_end
    if end < len(text):
        while end > start and not text[end - 1].isspace():
            end -= 1

        # fallback if no whitespace is found
        if end == start:
            end = approximate_end
    return end


class ApproximateFixedSizeSplitter(TextSplitter):
    """Text splitter which splits the input text into approximate fixed size chunks with
       optional overlap, avoiding cutting words.

    Args:
        chunk_size (int): The number of characters in each chunk.
        chunk_overlap (int): The number of characters from the previous chunk to overlap
                             with each chunk. Must be less than `chunk_size`.

    """
    @validate_call
    def __init__(self, chunk_size: int = 4000, chunk_overlap: int = 200) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @validate_call
    async def run(self, text: str) -> TextChunks:
        """Splits a piece of text into chunks without cutting words in half.

        Args:
            text (str): The text to be split.

        Returns:
            TextChunks: A list of TextChunk objects with chunked text.
        """
        chunks = []
        index = 0

        step = self.chunk_size - self.chunk_overlap
        text_length = len(text)

        i = 0
        while i < text_length:
            # adjust chunk start
            start = _adjust_chunk_start(text, i)

            # adjust chunk end
            approximate_end = min(start + self.chunk_size, text_length)
            end = _adjust_chunk_end(text, start, approximate_end)

            chunk_text = text[start:end]
            chunks.append(TextChunk(text=chunk_text, index=index))
            index += 1

            i = max(start + step, end)

        return TextChunks(chunks=chunks)
