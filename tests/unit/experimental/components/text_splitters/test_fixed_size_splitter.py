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
from itertools import zip_longest

import pytest
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import (
    FixedSizeSplitter,
)
from neo4j_graphrag.experimental.components.types import TextChunk


@pytest.mark.asyncio
async def test_split_text_no_overlap() -> None:
    text = "may thy knife chip and shatter"
    chunk_size = 5
    chunk_overlap = 0
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap)
    chunks = await splitter.run(text)
    expected_chunks = [
        TextChunk(text="may t", index=0),
        TextChunk(text="hy kn", index=1),
        TextChunk(text="ife c", index=2),
        TextChunk(text="hip a", index=3),
        TextChunk(text="nd sh", index=4),
        TextChunk(text="atter", index=5),
    ]
    for actual, expected in zip_longest(chunks.chunks, expected_chunks):
        assert actual.text == expected.text
        assert actual.index == expected.index
        assert expected.uid is not None


@pytest.mark.asyncio
async def test_split_text_with_overlap() -> None:
    text = "may thy knife chip and shatter"
    chunk_size = 10
    chunk_overlap = 2
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap)
    chunks = await splitter.run(text)
    expected_chunks = [
        TextChunk(text="may thy kn", index=0),
        TextChunk(text="knife chip", index=1),
        TextChunk(text="ip and sha", index=2),
        TextChunk(text="hatter", index=3),
    ]
    for actual, expected in zip_longest(chunks.chunks, expected_chunks):
        assert actual.text == expected.text
        assert actual.index == expected.index
        assert expected.uid is not None


@pytest.mark.asyncio
async def test_split_text_empty_string() -> None:
    text = ""
    chunk_size = 5
    chunk_overlap = 1
    splitter = FixedSizeSplitter(chunk_size, chunk_overlap)
    chunks = await splitter.run(text)
    assert chunks.chunks == []


def test_invalid_chunk_overlap() -> None:
    with pytest.raises(ValueError) as excinfo:
        FixedSizeSplitter(5, 5)
    assert "chunk_overlap must be strictly less than chunk_size" in str(excinfo)
